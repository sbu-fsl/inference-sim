package cmd

import (
	"encoding/csv"
	"encoding/json"
	"os"
	"strconv"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/energy"
)

// Energy CLI flags, shared by run and replay. Off unless --energy-config is given; the report
// and trace are files, so stdout is unchanged either way (INV-6, INV-13).
var (
	energyConfigPath string
	energyReportPath string
	energyTracePath  string
	energyFileConfig energy.Config // the loaded file, for the report's PUE / carbon terms
)

func registerEnergyFlags(cmd *cobra.Command) {
	cmd.Flags().StringVar(&energyConfigPath, "energy-config", "",
		"Strict-YAML GPU power model (model: tdp | utilization | resource_share); enables energy metering. See docs/guide/energy.md.")
	cmd.Flags().StringVar(&energyReportPath, "energy-report", "",
		"File to write the energy report (JSON). Requires --energy-config.")
	cmd.Flags().StringVar(&energyTracePath, "energy-trace", "",
		"File to write one CSV row per executed step (per-resource busy µs, watts). Requires --energy-config.")
}

// energySimConfig loads --energy-config for an instance of tp GPUs, or returns nil when energy
// is off. Refuses the configurations the meter cannot see in full rather than under-count (R1).
func energySimConfig(tp int) *sim.EnergyConfig {
	if energyConfigPath == "" {
		if energyReportPath != "" || energyTracePath != "" {
			logrus.Fatalf("--energy-report and --energy-trace require --energy-config")
		}
		return nil
	}
	if dataParallelism > 1 {
		logrus.Fatalf("--energy-config: a data-parallel group steps its ranks together " +
			"(sim/engine_driver_group.go), which the energy meter does not see yet")
	}
	if scenarioDisaggregated || kernelPDOpened != nil || prefillInstances > 0 || decodeInstances > 0 {
		logrus.Fatalf("--energy-config: P/D disaggregated deployments are not supported yet " +
			"(pools may differ in GPU count and the KV transfer is unpriced)")
	}
	model, c, err := energy.Load(energyConfigPath, tp)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	energyFileConfig = c
	return &sim.EnergyConfig{Model: model, IdleW: c.IdleW, GPUs: tp, Trace: energyTracePath != "",
		NeedsBreakdown: c.Model == "resource_share"}
}

type energyInstance struct {
	ID            string  `json:"id"`
	BusyJ         float64 `json:"busy_j"`
	IdleJ         float64 `json:"idle_j"`
	BusyS         float64 `json:"busy_s"`
	PrefillTokens int64   `json:"prefill_tokens"`
	DecodeTokens  int64   `json:"decode_tokens"`
}

type energyCluster struct {
	GPUJ            float64 `json:"gpu_j"`
	BusyJ           float64 `json:"busy_j"`
	IdleJ           float64 `json:"idle_j"`
	AvgWPerGPU      float64 `json:"avg_w_per_gpu"`
	AvgBusyWPerGPU  float64 `json:"avg_busy_w_per_gpu"`
	JPerOutputToken float64 `json:"j_per_output_token"`
	JPerRequest     float64 `json:"j_per_request"`
	GPUWh           float64 `json:"gpu_wh"`
	FacilityWh      float64 `json:"facility_wh"`
	GCO2            float64 `json:"g_co2"`
}

type energyReport struct {
	Model           string           `json:"model"`
	GPUsPerInstance int              `json:"gpus_per_instance"`
	PUE             float64          `json:"pue"`
	CarbonGPerKWh   float64          `json:"carbon_g_per_kwh"`
	SimSeconds      float64          `json:"sim_seconds"`
	Instances       []energyInstance `json:"instances"`
	Cluster         energyCluster    `json:"cluster"`
}

// writeEnergyOutputs writes --energy-report and --energy-trace after a run. No-op when off.
func writeEnergyOutputs(cs *cluster.ClusterSimulator, agg *sim.Metrics) {
	if energyConfigPath == "" {
		return
	}
	end := agg.SimEndedTime
	for _, inst := range cs.Instances() {
		if m := inst.EnergyMeter(); m != nil && m.Pipelined() {
			logrus.Fatalf("--energy-config: the engine split steps by pipeline stage (pp > 1), " +
				"which the energy meter does not price per stage yet; no energy report written")
		}
	}
	r := energyReport{Model: energyFileConfig.Model, PUE: energyFileConfig.PUE,
		CarbonGPerKWh: energyFileConfig.CarbonGPerKWh, SimSeconds: float64(end) / 1e6}
	var gpus int
	var busyUsGPU float64
	for _, inst := range cs.Instances() {
		m := inst.EnergyMeter()
		if m == nil {
			continue
		}
		r.GPUsPerInstance = m.GPUs()
		gpus += m.GPUs()
		busyUsGPU += float64(m.BusyUs() * int64(m.GPUs()))
		r.Instances = append(r.Instances, energyInstance{ID: string(inst.ID()), BusyJ: m.BusyJ(),
			IdleJ: m.IdleJ(end), BusyS: float64(m.BusyUs()) / 1e6,
			PrefillTokens: m.PrefillTokens(), DecodeTokens: m.DecodeTokens()})
		r.Cluster.BusyJ += m.BusyJ()
		r.Cluster.IdleJ += m.IdleJ(end)
	}
	c := &r.Cluster
	c.GPUJ = c.BusyJ + c.IdleJ
	if end > 0 && gpus > 0 {
		c.AvgWPerGPU = c.GPUJ / (float64(end) / 1e6) / float64(gpus)
	}
	if busyUsGPU > 0 {
		c.AvgBusyWPerGPU = c.BusyJ / (busyUsGPU / 1e6)
	}
	if agg.TotalOutputTokens > 0 {
		c.JPerOutputToken = c.GPUJ / float64(agg.TotalOutputTokens)
	}
	if agg.CompletedRequests > 0 {
		c.JPerRequest = c.GPUJ / float64(agg.CompletedRequests)
	}
	c.GPUWh = c.GPUJ / 3600
	c.FacilityWh = c.GPUWh * r.PUE
	c.GCO2 = c.FacilityWh / 1000 * r.CarbonGPerKWh

	if energyReportPath != "" {
		b, err := json.MarshalIndent(r, "", "  ")
		if err != nil {
			logrus.Fatalf("energy report: %v", err)
		}
		if err := os.WriteFile(energyReportPath, append(b, '\n'), 0o644); err != nil {
			logrus.Fatalf("energy report: %v", err)
		}
	}
	if energyTracePath != "" {
		writeEnergyTrace(cs)
	}
	logrus.Infof("energy (%s): %.1f J GPU, %.1f W/GPU avg, %.4f J/output-token",
		r.Model, c.GPUJ, c.AvgWPerGPU, c.JPerOutputToken)
}

func writeEnergyTrace(cs *cluster.ClusterSimulator) {
	f, err := os.Create(energyTracePath)
	if err != nil {
		logrus.Fatalf("energy trace: %v", err)
	}
	w := csv.NewWriter(f)
	head := []string{"instance", "start_us", "dur_us", "batch", "prefill_tokens", "decode_tokens"}
	for _, res := range sim.EnergyResources {
		head = append(head, string(res)+"_us")
	}
	_ = w.Write(append(head, "watts_per_gpu"))
	i64 := func(v int64) string { return strconv.FormatInt(v, 10) }
	for _, inst := range cs.Instances() {
		m := inst.EnergyMeter()
		if m == nil {
			continue
		}
		for _, s := range m.Trace() {
			row := []string{string(inst.ID()), i64(s.StartUs), i64(s.DurUs), strconv.Itoa(s.Batch),
				strconv.Itoa(s.Prefill), strconv.Itoa(s.Decode)}
			for _, b := range s.BusyUs {
				row = append(row, i64(b))
			}
			_ = w.Write(append(row, strconv.FormatFloat(s.Watts, 'f', 3, 64)))
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		logrus.Fatalf("energy trace: %v", err)
	}
	if err := f.Close(); err != nil {
		logrus.Fatalf("energy trace: %v", err)
	}
}

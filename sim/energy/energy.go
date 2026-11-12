// Package energy holds BLIS's GPU power models and their config loader.
//
// A power model maps one executed step (sim.StepSample) to watts per GPU; sim.EnergyMeter
// integrates it over the run. Adding a model: define a type with StepPower, add its YAML block
// to Config, and one case to build. Guide: docs/guide/energy.md; evidence and the selection
// gate: experiments/energy/README.md.
package energy

import (
	"bytes"
	"fmt"
	"math"
	"os"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/sim"
)

// TDP draws a constant power whenever a step runs: the "assume rated power" baseline that
// ML.ENERGY (arXiv:2505.06371) and Wilkins et al. (arXiv:2603.18383) show overestimates.
type TDP struct{ Watts float64 }

func (t TDP) StepPower(sim.StepSample) float64 { return t.Watts }

// Utilization is Vidur-Energy's model (Özcan et al., arXiv:2507.11417, Eq. 1–2):
//
//	P = IdleW + (MaxW − IdleW) · min(1, MFU/MFUSat)^Gamma
//	MFU = FLOPsPerTokenPerGPU · tokens / (PeakTFLOPS·1e12 · step seconds)
//
// FLOPs per token is the Kaplan 2·N_active approximation; attention FLOPs are not counted.
type Utilization struct {
	IdleW, MaxW, MFUSat, Gamma, PeakTFLOPS, FLOPsPerTokenPerGPU float64
}

// MFU is the step's model-FLOPs utilization.
func (u Utilization) MFU(s sim.StepSample) float64 {
	sec := s.Duration.Seconds()
	if sec <= 0 {
		return 0
	}
	return u.FLOPsPerTokenPerGPU * float64(s.PrefillTokens+s.DecodeTokens) / (u.PeakTFLOPS * 1e12 * sec)
}

func (u Utilization) StepPower(s sim.StepSample) float64 {
	x := math.Min(1, u.MFU(s)/u.MFUSat)
	return u.IdleW + (u.MaxW-u.IdleW)*math.Pow(x, u.Gamma)
}

// ResourceShare prices a step as the time-weighted mix of per-resource power levels, using
// the latency kernel's per-resource busy time: P = Σ_r (t_r / Σ t) · W[r]. A compute-bound
// prefill lands near W[sm], a memory-bound decode near W[hbm] (Patel et al., ASPLOS'24).
// A step with no breakdown is charged W[host].
type ResourceShare struct{ W map[kernel.Resource]float64 }

func (r ResourceShare) StepPower(s sim.StepSample) float64 {
	var total, weighted float64
	for _, res := range sim.EnergyResources { // fixed order: deterministic rounding
		t := s.Busy[res].Seconds()
		total += t
		weighted += t * r.W[res]
	}
	if total <= 0 {
		return r.W[kernel.ResourceHost]
	}
	return weighted / total
}

// Config is the strict-YAML energy file (--energy-config).
type Config struct {
	Model         string    `yaml:"model"`
	IdleW         float64   `yaml:"idle_w"`
	PUE           float64   `yaml:"pue"`
	CarbonGPerKWh float64   `yaml:"carbon_g_per_kwh"`
	TDP           *tdpYAML  `yaml:"tdp"`
	Utilization   *utilYAML `yaml:"utilization"`
	ResourceShare *rsYAML   `yaml:"resource_share"`
}

type tdpYAML struct {
	Watts float64 `yaml:"watts"`
}

type utilYAML struct {
	// FloorW is the curve's P_idle (power at MFU 0 while a step runs). Optional: it defaults to
	// idle_w, as in Vidur-Energy, but a fitted floor is an "active" power, not what an idle GPU
	// draws between steps, so the two can be set apart.
	FloorW        *float64 `yaml:"floor_w"`
	MaxW          float64  `yaml:"max_w"`
	MFUSat        float64  `yaml:"mfu_sat"`
	Gamma         float64  `yaml:"gamma"`
	PeakTFLOPS    float64  `yaml:"peak_tflops"`
	ActiveParamsB float64  `yaml:"active_params_b"`
}

type rsYAML struct {
	SMW     float64 `yaml:"sm_w"`
	HBMW    float64 `yaml:"hbm_w"`
	NVLinkW float64 `yaml:"nvlink_w"`
	NICW    float64 `yaml:"nic_w"`
	PCIeW   float64 `yaml:"pcie_w"`
	CopyW   float64 `yaml:"copy_engine_w"`
	HostW   float64 `yaml:"host_w"`
}

// Load reads and validates an energy config; tp is the instance's tensor-parallel width
// (the Utilization model's per-GPU FLOPs divide by it).
func Load(path string, tp int) (sim.PowerModel, Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, Config{}, err
	}
	var c Config
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	if err := dec.Decode(&c); err != nil {
		return nil, Config{}, fmt.Errorf("energy config %s: %w", path, err)
	}
	if c.PUE == 0 {
		c.PUE = 1
	}
	m, err := build(c, tp)
	if err != nil {
		return nil, Config{}, fmt.Errorf("energy config %s: %w", path, err)
	}
	return m, c, nil
}

func build(c Config, tp int) (sim.PowerModel, error) {
	if tp < 1 {
		return nil, fmt.Errorf("tensor-parallel width %d < 1", tp)
	}
	if err := nonNeg("idle_w", c.IdleW, "pue", c.PUE, "carbon_g_per_kwh", c.CarbonGPerKWh); err != nil {
		return nil, err
	}
	if c.PUE < 1 {
		return nil, fmt.Errorf("pue %v < 1", c.PUE)
	}
	switch c.Model {
	case "tdp":
		if c.TDP == nil {
			return nil, fmt.Errorf("model tdp needs a tdp: block")
		}
		return TDP{Watts: c.TDP.Watts}, nonNeg("tdp.watts", c.TDP.Watts)
	case "utilization":
		u := c.Utilization
		if u == nil {
			return nil, fmt.Errorf("model utilization needs a utilization: block")
		}
		if err := nonNeg("max_w", u.MaxW, "gamma", u.Gamma); err != nil {
			return nil, err
		}
		if !(u.MFUSat > 0 && u.MFUSat <= 1) {
			return nil, fmt.Errorf("utilization.mfu_sat %v not in (0, 1]", u.MFUSat)
		}
		if !(u.PeakTFLOPS > 0 && u.ActiveParamsB > 0) {
			return nil, fmt.Errorf("utilization needs peak_tflops > 0 and active_params_b > 0")
		}
		floor := c.IdleW
		if u.FloorW != nil {
			floor = *u.FloorW
			if err := nonNeg("utilization.floor_w", floor); err != nil {
				return nil, err
			}
		}
		return Utilization{IdleW: floor, MaxW: u.MaxW, MFUSat: u.MFUSat, Gamma: u.Gamma,
			PeakTFLOPS: u.PeakTFLOPS, FLOPsPerTokenPerGPU: 2 * u.ActiveParamsB * 1e9 / float64(tp)}, nil
	case "resource_share":
		r := c.ResourceShare
		if r == nil {
			return nil, fmt.Errorf("model resource_share needs a resource_share: block")
		}
		if err := nonNeg("sm_w", r.SMW, "hbm_w", r.HBMW, "nvlink_w", r.NVLinkW, "nic_w", r.NICW,
			"pcie_w", r.PCIeW, "copy_engine_w", r.CopyW, "host_w", r.HostW); err != nil {
			return nil, err
		}
		return ResourceShare{W: map[kernel.Resource]float64{
			kernel.ResourceSM: r.SMW, kernel.ResourceHBM: r.HBMW, kernel.ResourceNVLink: r.NVLinkW,
			kernel.ResourceNIC: r.NICW, kernel.ResourcePCIe: r.PCIeW,
			kernel.ResourceCopyEngine: r.CopyW, kernel.ResourceHost: r.HostW,
		}}, nil
	default:
		return nil, fmt.Errorf("unknown model %q (want tdp, utilization or resource_share)", c.Model)
	}
}

// nonNeg checks name/value pairs are finite and >= 0.
func nonNeg(kv ...any) error {
	for i := 0; i < len(kv); i += 2 {
		v := kv[i+1].(float64)
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("%s = %v; must be finite and >= 0", kv[i], v)
		}
	}
	return nil
}

package energy

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"

	"github.com/inference-sim/inference-sim/sim"
)

func us(n int) time.Duration { return time.Duration(n) * time.Microsecond }

func TestUtilization_Curve(t *testing.T) {
	u := Utilization{IdleW: 60, MaxW: 700, MFUSat: 0.45, Gamma: 0.7, PeakTFLOPS: 1, FLOPsPerTokenPerGPU: 1e6}
	at := func(tokens int) float64 { // 1 ms step: MFU = tokens·1e6 / 1e9 = tokens/1000
		return u.StepPower(sim.StepSample{Duration: us(1000), DecodeTokens: tokens})
	}
	if at(0) != 60 {
		t.Errorf("MFU 0: %v W, want idle 60", at(0))
	}
	if at(450) != 700 || at(900) != 700 {
		t.Errorf("MFU >= MFU_sat must saturate at 700: %v %v", at(450), at(900))
	}
	if want := 60 + 640*math.Pow(0.5, 0.7); math.Abs(at(225)-want) > 1e-9 {
		t.Errorf("MFU half of sat: %v, want %v", at(225), want)
	}
	for k := 1; k < 450; k += 50 {
		if at(k) > at(k+50) {
			t.Fatalf("not monotone at %d tokens", k)
		}
	}
}

func TestResourceShare_Laws(t *testing.T) {
	r := ResourceShare{W: map[kernel.Resource]float64{
		kernel.ResourceSM: 600, kernel.ResourceHBM: 350, kernel.ResourceHost: 60}}
	cases := []struct {
		name string
		busy map[kernel.Resource]time.Duration
		want float64
	}{
		{"pure compute", map[kernel.Resource]time.Duration{kernel.ResourceSM: us(10)}, 600},
		{"pure memory", map[kernel.Resource]time.Duration{kernel.ResourceHBM: us(10)}, 350},
		{"no breakdown", nil, 60},
		{"25/75 sm/hbm", map[kernel.Resource]time.Duration{kernel.ResourceSM: us(25), kernel.ResourceHBM: us(75)}, 412.5},
	}
	for _, c := range cases {
		got := r.StepPower(sim.StepSample{Busy: c.busy})
		if math.Abs(got-c.want) > 1e-9 {
			t.Errorf("%s: %v W, want %v", c.name, got, c.want)
		}
		if got < 60 || got > 600 {
			t.Errorf("%s: %v W outside [min W, max W]", c.name, got)
		}
	}
}

func write(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "e.yaml")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// The shipped example configs load, and the utilization model divides FLOPs by TP.
func TestLoad_Examples(t *testing.T) {
	for _, f := range []string{"h100-resource-share.yaml", "h100-utilization.yaml", "h100-tdp.yaml"} {
		m, c, err := Load(filepath.Join("..", "..", "examples", "energy", f), 4)
		if err != nil || m == nil || c.PUE != 1 {
			t.Fatalf("%s: %v", f, err)
		}
		if u, ok := m.(Utilization); ok && u.FLOPsPerTokenPerGPU != 2*8e9/4 {
			t.Errorf("FLOPs/token/GPU %v, want 4e9", u.FLOPsPerTokenPerGPU)
		}
	}
}

// floor_w sets the curve's MFU-0 power without touching the between-step idle power.
func TestLoad_UtilizationFloor(t *testing.T) {
	m, c, err := Load(write(t, "model: utilization\nidle_w: 60\nutilization: {floor_w: 440, max_w: 690, mfu_sat: 0.15, gamma: 0.8, peak_tflops: 989.5, active_params_b: 8}\n"), 1)
	if err != nil {
		t.Fatal(err)
	}
	if c.IdleW != 60 || m.StepPower(sim.StepSample{Duration: us(1000)}) != 440 {
		t.Fatalf("idle %v, MFU-0 power %v; want 60 and 440", c.IdleW, m.StepPower(sim.StepSample{Duration: us(1000)}))
	}
}

func TestLoad_Rejects(t *testing.T) {
	bad := map[string]string{
		"unknown key":     "model: tdp\ntdp: {watts: 700}\nwat: 1\n",
		"unknown model":   "model: magic\n",
		"missing block":   "model: resource_share\n",
		"negative watts":  "model: tdp\ntdp: {watts: -1}\n",
		"negative idle":   "model: tdp\nidle_w: -5\ntdp: {watts: 1}\n",
		"pue below one":   "model: tdp\npue: 0.5\ntdp: {watts: 1}\n",
		"mfu_sat zero":    "model: utilization\nutilization: {max_w: 700, mfu_sat: 0, gamma: 1, peak_tflops: 1, active_params_b: 8}\n",
		"mfu_sat above 1": "model: utilization\nutilization: {max_w: 700, mfu_sat: 1.5, gamma: 1, peak_tflops: 1, active_params_b: 8}\n",
		"no params":       "model: utilization\nutilization: {max_w: 700, mfu_sat: 0.4, gamma: 1, peak_tflops: 1}\n",
		"negative sm":     "model: resource_share\nresource_share: {sm_w: -1}\n",
	}
	for name, body := range bad {
		if _, _, err := Load(write(t, body), 1); err == nil {
			t.Errorf("%s: accepted", name)
		} else if !strings.Contains(err.Error(), "energy config") {
			t.Errorf("%s: error %q does not name the file", name, err)
		}
	}
}

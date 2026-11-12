package sim

import (
	"math"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

type constPower float64

func (c constPower) StepPower(StepSample) float64 { return float64(c) }

type spyPower struct{ last StepSample }

func (s *spyPower) StepPower(x StepSample) float64 { s.last = x; return 0 }

// plainEngine is an engine.Engine with no breakdown; breakdownEngine adds one. Only the type
// assertion is exercised, so the embedded nil interface is never called.
type plainEngine struct{ engine.Engine }

type breakdownEngine struct {
	engine.Engine
	per map[kernel.Resource]time.Duration
}

func (b breakdownEngine) StepEstimate(engine.Batch) kernel.StepEstimate {
	return kernel.StepEstimate{PerResource: b.per}
}

func TestNewEnergyMeter_NilAndInvalid(t *testing.T) {
	if m, err := NewEnergyMeter(nil, plainEngine{}); m != nil || err != nil {
		t.Fatal("nil config built a meter")
	}
	for _, c := range []EnergyConfig{{GPUs: 1}, {Model: constPower(1)}, {Model: constPower(1), GPUs: 1, IdleW: -1}} {
		if _, err := NewEnergyMeter(&c, plainEngine{}); err == nil {
			t.Errorf("%+v accepted", c)
		}
	}
	if _, err := NewEnergyMeter(&EnergyConfig{Model: constPower(1), GPUs: 1, NeedsBreakdown: true}, plainEngine{}); err == nil {
		t.Error("a model needing a breakdown accepted an engine without one")
	}
}

// Energy is power × time × GPUs, additive over steps; idle fills the rest of the horizon.
func TestEnergyMeter_EnergyLaws(t *testing.T) {
	em, err := NewEnergyMeter(&EnergyConfig{Model: constPower(500), IdleW: 60, GPUs: 2}, plainEngine{})
	if err != nil {
		t.Fatal(err)
	}
	em.Record(engine.Batch{}, 0, 1000)
	em.Record(engine.Batch{}, 5000, 1000)
	if math.Abs(em.BusyJ()-2.0) > 1e-12 || em.BusyUs() != 2000 {
		t.Fatalf("busy %v J over %d µs, want 2 J over 2000 µs", em.BusyJ(), em.BusyUs())
	}
	if got := em.IdleJ(10_000); math.Abs(got-0.96) > 1e-12 {
		t.Fatalf("idle %v J, want 0.96", got)
	}
	if em.IdleJ(1000) != 0 {
		t.Fatal("idle energy not clamped at zero")
	}
}

// A request whose prompt is not yet computed runs prefill; one past it decodes. The engine's
// breakdown reaches the power model when it exposes one.
func TestEnergyMeter_SampleContents(t *testing.T) {
	spy := &spyPower{}
	eng := breakdownEngine{per: map[kernel.Resource]time.Duration{kernel.ResourceSM: 3 * time.Microsecond}}
	em, err := NewEnergyMeter(&EnergyConfig{Model: spy, GPUs: 1, Trace: true, NeedsBreakdown: true}, eng)
	if err != nil {
		t.Fatal(err)
	}
	b := engine.Batch{Reqs: []engine.Scheduled{
		{ID: "p", Tokens: 64, Computed: 10, PromptLen: 100},
		{ID: "d", Tokens: 1, Computed: 120, PromptLen: 100},
	}}
	em.Record(b, 42, 5)
	s := spy.last
	if s.PrefillTokens != 64 || s.DecodeTokens != 1 || s.BatchSize != 2 || s.Duration != 5*time.Microsecond {
		t.Fatalf("sample %+v", s)
	}
	if s.Busy[kernel.ResourceSM] != 3*time.Microsecond {
		t.Fatal("breakdown did not reach the power model")
	}
	if tr := em.Trace(); len(tr) != 1 || tr[0].StartUs != 42 || tr[0].BusyUs[0] != 3 || tr[0].Prefill != 64 {
		t.Fatalf("trace %+v", tr)
	}
}

func TestEnergyMeter_PanicsOnBadPower(t *testing.T) {
	em, _ := NewEnergyMeter(&EnergyConfig{Model: constPower(math.NaN()), GPUs: 1}, plainEngine{})
	defer func() {
		if recover() == nil {
			t.Fatal("NaN power did not panic")
		}
	}()
	em.Record(engine.Batch{}, 0, 1)
}

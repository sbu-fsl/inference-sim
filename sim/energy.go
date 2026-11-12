package sim

import (
	"fmt"
	"math"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// EnergyResources is the fixed order every per-resource sum runs in. Summing a map in
// iteration order would make the energy report's float rounding run-dependent (INV-6).
var EnergyResources = [...]kernel.Resource{
	kernel.ResourceSM, kernel.ResourceHBM, kernel.ResourceNVLink, kernel.ResourceNIC,
	kernel.ResourcePCIe, kernel.ResourceCopyEngine, kernel.ResourceHost,
}

// StepSample is everything a power model may read about one executed step.
type StepSample struct {
	Duration time.Duration
	// Busy is the per-resource busy time of the step (kernel.StepEstimate.PerResource); nil
	// when the engine exposes no breakdown (StepBreakdown).
	Busy          map[kernel.Resource]time.Duration
	PrefillTokens int
	DecodeTokens  int
	BatchSize     int
}

// PowerModel returns the average electrical power, in watts per GPU, drawn while a step runs.
// Implementations live in sim/energy; a new model is a new type there.
type PowerModel interface {
	StepPower(s StepSample) float64
}

// EnergyConfig turns on an instance's energy metering. A nil *EnergyConfig on
// EngineDriverConfig is inert: no meter is built and output is byte-identical (INV-6).
type EnergyConfig struct {
	Model PowerModel
	IdleW float64 // per-GPU power while the instance runs no step
	GPUs  int     // GPUs one instance occupies (its tensor-parallel width)
	Trace bool    // keep one EnergyStep per executed step
	// NeedsBreakdown makes the meter refuse an engine that exposes no StepBreakdown, for a
	// model (resource_share) that reads per-resource busy time.
	NeedsBreakdown bool
}

// StepBreakdown is the optional engine surface the meter reads per-resource busy time from.
// The blis-vllm engine does not expose it yet, so a model that needs it is refused there.
type StepBreakdown interface {
	StepEstimate(b engine.Batch) kernel.StepEstimate
}

// EnergyStep is one executed step in the meter's trace. StartUs is the step's start on the
// simulation clock.
type EnergyStep struct {
	StartUs, DurUs         int64
	Batch, Prefill, Decode int
	BusyUs                 [len(EnergyResources)]int64
	Watts                  float64
}

// EnergyMeter accumulates one instance's GPU energy from the steps its EngineDriver prices.
type EnergyMeter struct {
	cfg       EnergyConfig
	breakdown StepBreakdown // nil when the engine exposes none
	busyJ     float64
	busyUs    int64
	prefill   int64
	decode    int64
	pipelined bool
	trace     []EnergyStep
}

// NewEnergyMeter builds a meter for eng, or returns nil for a nil cfg. Errors are returned, not
// panicked, so the CLI can name them (R1).
func NewEnergyMeter(cfg *EnergyConfig, eng engine.Engine) (*EnergyMeter, error) {
	if cfg == nil {
		return nil, nil
	}
	if cfg.Model == nil || cfg.GPUs < 1 || cfg.IdleW < 0 || math.IsNaN(cfg.IdleW) || math.IsInf(cfg.IdleW, 0) {
		return nil, fmt.Errorf("energy meter: invalid config (model nil=%v, gpus %d, idle %v W)",
			cfg.Model == nil, cfg.GPUs, cfg.IdleW)
	}
	m := &EnergyMeter{cfg: *cfg}
	m.breakdown, _ = eng.(StepBreakdown)
	if cfg.NeedsBreakdown && m.breakdown == nil {
		return nil, fmt.Errorf("energy meter: the power model needs per-resource busy time, " +
			"which this engine does not expose (sim.StepBreakdown)")
	}
	return m, nil
}

// Record charges one step of batch b that runs [startUs, startUs+durUs).
func (e *EnergyMeter) Record(b engine.Batch, startUs, durUs int64) {
	s := StepSample{Duration: time.Duration(durUs) * time.Microsecond, BatchSize: len(b.Reqs)}
	for _, r := range b.Reqs {
		if r.Computed < r.PromptLen { // a prefill chunk: the prompt is not yet fully computed
			s.PrefillTokens += r.Tokens
		} else {
			s.DecodeTokens += r.Tokens
		}
	}
	if e.breakdown != nil {
		s.Busy = e.breakdown.StepEstimate(b).PerResource
	}
	w := e.cfg.Model.StepPower(s)
	if math.IsNaN(w) || math.IsInf(w, 0) || w < 0 {
		panic(fmt.Sprintf("PowerModel.StepPower = %v; the contract is finite and >= 0", w))
	}
	if e.cfg.Trace {
		st := EnergyStep{StartUs: startUs, DurUs: durUs, Batch: s.BatchSize,
			Prefill: s.PrefillTokens, Decode: s.DecodeTokens, Watts: w}
		for i, r := range EnergyResources {
			st.BusyUs[i] = s.Busy[r].Microseconds()
		}
		e.trace = append(e.trace, st)
	}
	e.busyJ += w * float64(e.cfg.GPUs) * float64(durUs) / 1e6
	e.busyUs += durUs
	e.prefill += int64(s.PrefillTokens)
	e.decode += int64(s.DecodeTokens)
}

// NotePipelined marks that the engine split a step by pipeline stage, which the meter does not
// price per stage; the CLI refuses such a run when it ends rather than report it (R1).
func (e *EnergyMeter) NotePipelined() { e.pipelined = true }

// Pipelined reports whether a pipelined step was seen.
func (e *EnergyMeter) Pipelined() bool { return e.pipelined }

// BusyJ is the energy (J, all of the instance's GPUs) drawn while running steps.
func (e *EnergyMeter) BusyJ() float64 { return e.busyJ }

// BusyUs is the instance's total step time in µs.
func (e *EnergyMeter) BusyUs() int64 { return e.busyUs }

// PrefillTokens and DecodeTokens are the forward-pass tokens the instance ran.
func (e *EnergyMeter) PrefillTokens() int64 { return e.prefill }
func (e *EnergyMeter) DecodeTokens() int64  { return e.decode }

// IdleJ is the idle energy (J) from time 0 to endUs outside the instance's steps.
func (e *EnergyMeter) IdleJ(endUs int64) float64 {
	return e.cfg.IdleW * float64(e.cfg.GPUs) * float64(max(0, endUs-e.busyUs)) / 1e6
}

// GPUs is the GPU count the meter charges per instance.
func (e *EnergyMeter) GPUs() int { return e.cfg.GPUs }

// Trace is the per-step record (empty unless EnergyConfig.Trace).
func (e *EnergyMeter) Trace() []EnergyStep { return e.trace }

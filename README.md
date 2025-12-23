# Scaffolding Asymmetry

**Task-specific scaffolding effects in large language models: Evidence for dissociable generation and verification deficits.**

---

## The Finding

Self-monitoring scaffolds dramatically improve verification (+28%, p<.001), but **all scaffolding degrades generation**. Crossed scaffold-task pairings hurt worst.

| | Self-Monitor | Constraint |
|---|---|---|
| **Arithmetic (Verification)** | ✅ +28% | ❌ -10% |
| **Logic (Generation)** | ❌ -32% | ❌ -30% |

*Claude Sonnet 4, N=50 per cell*

---

## Why This Matters

LLMs have **dissociable deficits**:
- **Verification deficit**: Can compute but can't hold-and-check
- **Generation deficit**: Can't satisfy constraints while constructing output

Scaffolding isn't uniformly helpful. It has **cognitive overhead**. When the overhead addresses the task's bottleneck → benefit. When it doesn't → cost.

---

## The Experiment

### Tasks

**Arithmetic Verification** (verification deficit):
> "Is 17 × 24 = 408? Answer only Yes or No."

**Hillary Logic™** (generation deficit):
> "If the prompt contains BOTH a 3-letter word AND a 6-letter word, respond with EXACTLY 4 words..."

Hillary Logic uses **arbitrary conditional rules** that can't be pattern-matched from training. Must actually:
1. Check condition A (hold result)
2. Check condition B (hold result)
3. Apply correct rule
4. Generate compliant response

### Scaffolds

**Self-Monitor**: "Compute, hold, check, then answer."

**Constraint**: "List all constraints, check each one, then answer."

---

## Run It

```bash
pip install anthropic pandas scipy numpy

# Full experiment (both tasks, both models)
python scaffolding_experiment.py --n 50 --provider anthropic --output results_claude
python scaffolding_experiment.py --n 50 --provider openai --model gpt-4o --output results_gpt4o

# Just logic tasks
python scaffolding_experiment.py --n 50 --logic

# Just arithmetic tasks  
python scaffolding_experiment.py --n 50 --arithmetic

# Quick test
python scaffolding_experiment.py --n 10 --logic
```

---

## Results

### Claude Sonnet 4

| Task | Baseline | Self-Monitor | Constraint |
|------|----------|--------------|------------|
| Arithmetic | 70% | **98%** | 60% |
| Logic | 64% | 32% | 34% |

### GPT-4o

| Task | Baseline | Self-Monitor | Constraint |
|------|----------|--------------|------------|
| Arithmetic | 88% | 90% | 66% |
| Logic | 66% | 60% | 58% |

### Key Stats

- Self-monitor on arithmetic (Claude): **+28%**, t=4.09, p<.001
- Matched scaffolds: -1% to -3% (small cost)
- Crossed scaffolds: -14% to -21% (large cost)

---

## Interpretation

**Cognitive Overhead Account**: Scaffolding imposes processing demands.

- When demands address the bottleneck → benefit outweighs cost
- When demands miss the bottleneck → cost dominates
- When demands address the *wrong* bottleneck → largest cost

This explains why:
- ✅ Self-monitor helps verification (provides the hold-and-check structure)
- ❌ All scaffolds hurt generation (adds overhead to capacity-limited process)
- ❌❌ Crossed pairings hurt most (wrong overhead, no benefit)

---

## Files

| File | Description |
|------|-------------|
| `scaffolding_experiment.py` | Full experiment code |
| `scaffolding_asymmetry_paper.md` | Publication-ready writeup |
| `results/` | Raw data and analysis |

---

## Part of the Abstraction-Intelligence Framework

👉 **[Main Repository](https://github.com/HillaryDanan/abstraction-intelligence)**

This experiment tests predictions from the **Abstraction Primitive Hypothesis**:
- [The Geometry of Self-Reference](https://github.com/HillaryDanan/geometry-self-reference)
- [Abstraction Stages Demo](https://github.com/HillaryDanan/abstraction-stages)

---

## Citation

```bibtex
@article{danan2025scaffolding,
  title={Task-Specific Scaffolding Effects in Large Language Models: 
         Evidence for Dissociable Generation and Verification Deficits},
  author={Danan, Hillary},
  year={2025},
  url={https://github.com/HillaryDanan/scaffolding-asymmetry}
}
```

---

## Author

**Hillary Danan, PhD** · Cognitive Neuroscience

---

*"The data surprised us. That's how you know it's real."*

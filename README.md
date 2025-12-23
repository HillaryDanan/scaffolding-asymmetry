# Scaffolding Asymmetry Experiment

**Testing predictions from "The Geometry of Self-Reference" (Danan, 2025)**

## The Prediction

The geometric framework predicts **task-specific scaffold effects**:

| Task Type | Known Deficit | Best Scaffold | Why |
|-----------|--------------|---------------|-----|
| Arithmetic Verification | Verification deficit | Self-monitoring | Helps HOLD computed value while checking |
| Logic Generation | Generation deficit | Constraint-checking | Helps CHECK constraints before committing |

**Crossed scaffolds should show smaller effects.**

This is Prediction 2 from the paper. The simpler "working memory" account predicts uniform effects. The geometric account predicts an **interaction**.

## Design

```
                    Scaffold Type
                    ─────────────────────────────────────
                    Baseline    Self-Monitor    Constraint
Task Type
─────────────────
Arithmetic           base_A      ↑↑ MATCHED      crossed
Verification

Logic                base_L      crossed         ↑↑ MATCHED
Generation
```

## Run

```bash
# Install dependencies
pip install anthropic pandas scipy numpy

# Run with Anthropic (default)
python scaffolding_experiment.py --n 50

# Run with specific model
python scaffolding_experiment.py --n 50 --model claude-3-5-sonnet-20241022

# Run with OpenAI
pip install openai
python scaffolding_experiment.py --n 50 --provider openai --model gpt-4

# Quick test run
python scaffolding_experiment.py --n 10
```

## Output

Results saved to `results/`:
- `raw_results.csv` — All trial data
- `analysis.json` — Statistical analysis

## Expected Results (If Prediction Correct)

```
### Interaction Test (Key Prediction) ###
  Mean improvement (matched scaffolds): +15-25%
  Mean improvement (crossed scaffolds): +0-5%
  
  >>> CONFIRMED: Matched scaffolds help more than crossed scaffolds <<<
```

## Interpreting Results

| Outcome | Interpretation |
|---------|----------------|
| Matched >> Crossed | ✓ Supports geometric framework |
| Matched ≈ Crossed | ✗ No interaction; simpler account suffices |
| All scaffolds help equally | ✗ Just "prompting helps"; no task-specificity |

## Citation

```bibtex
@article{danan2025geometry,
  title={The Geometry of Self-Reference: Information-Theoretic Foundations 
         for Self-State and Metacognitive Capacity},
  author={Danan, Hillary},
  year={2025},
  note={Working paper}
}
```

## Author

Based on framework by Hillary Danan, PhD
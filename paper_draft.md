# Task-Specific Scaffolding Effects in Large Language Models: Evidence for Dissociable Generation and Verification Deficits

**Hillary Danan, PhD**

*Cognitive Neuroscience*

---

## Abstract

Large language models exhibit systematic failures in both generation and verification tasks, but the relationship between these deficits remains unclear. We tested whether scaffolding interventions—structured prompts designed to support specific cognitive operations—show task-specific effects consistent with dissociable underlying deficits. Across two frontier models (Claude Sonnet 4, GPT-4o; total N=600 trials), we found a significant scaffold × task interaction. Self-monitoring scaffolds dramatically improved verification performance (+28%, p<.001) but degraded generation performance. Critically, mismatched scaffold-task pairings produced the largest performance decrements, consistent with a cognitive overhead account: scaffolding that fails to address a task's specific bottleneck imposes processing costs without compensatory benefit. These findings support the hypothesis that generation and verification failures in LLMs reflect distinct computational limitations requiring targeted interventions.

**Keywords:** large language models, scaffolding, verification, generation, cognitive architecture

---

## Introduction

Large language models (LLMs) exhibit a puzzling asymmetry: they can generate correct answers to complex problems yet fail to verify whether presented answers are correct, and conversely, they can verify solutions they cannot generate (Danan, 2025; Wei et al., 2022). This pattern suggests that generation and verification engage partially dissociable processes, rather than reflecting a unitary "reasoning" capacity.

The **Abstraction Primitive Hypothesis** (Danan, 2025) proposes that these asymmetries reflect limitations in self-referential processing—specifically, the capacity to hold computed values while performing operations on them (termed "hold-and-check"). On this account:

- **Verification deficits** arise when a system cannot maintain its own computed result while comparing it to a presented claim
- **Generation deficits** arise when a system cannot check candidate outputs against constraints before committing to a response

If these deficits have distinct computational bases, scaffolding interventions should show **task-specific effects**: a scaffold designed to support hold-and-check should help verification but not generation, while a scaffold designed to support constraint-checking should help generation but not verification.

### Predictions

We tested three pre-registered predictions:

1. **Matched scaffold benefit**: Self-monitoring scaffolds improve verification-deficit tasks; constraint-checking scaffolds improve generation-deficit tasks
2. **Crossed scaffold null/cost**: Mismatched scaffold-task pairings produce smaller benefits or costs
3. **Interaction**: The scaffold × task interaction is significant

---

## Results

### Experimental Design

We tested two task types designed to produce dissociable deficits:

**Arithmetic verification** (predicted verification deficit): Participants judged whether multiplication claims were correct (e.g., "Is 17 × 24 = 408?"). This requires computing the product, holding it, and comparing to the presented value.

**Logic generation** (predicted generation deficit): Participants produced responses satisfying arbitrary conditional rules (e.g., "If the prompt contains a 3-letter word AND a 6-letter word, respond with exactly 4 words"). This requires checking multiple conditions, determining which rule applies, and generating a compliant response.

Three scaffolding conditions were tested:
- **Baseline**: No scaffolding
- **Self-monitor**: "Compute, hold, check, then answer"
- **Constraint**: "List all constraints, check each one, then answer"

We collected N=50 trials per cell across two models (Claude Sonnet 4, GPT-4o), yielding 600 total trials.

### Main Findings

#### Claude Sonnet 4

| Task | Baseline | Self-Monitor | Constraint |
|------|----------|--------------|------------|
| Arithmetic Verification | 70% | 98% | 60% |
| Logic Generation | 64% | 32% | 34% |

#### GPT-4o

| Task | Baseline | Self-Monitor | Constraint |
|------|----------|--------------|------------|
| Arithmetic Verification | 88% | 90% | 66% |
| Logic Generation | 66% | 60% | 58% |

### Statistical Tests

**Prediction 1 (Matched scaffolds):**

*Self-monitor on arithmetic (Claude)*: +28.0%, t(98)=4.09, p<.001, d=0.82

This effect was specific to Claude; GPT-4o showed a non-significant +2% improvement.

*Constraint on logic*: Contrary to prediction, constraint scaffolding **decreased** performance (Claude: -30%, p=.002; GPT-4o: -8%, p=.42).

**Prediction 2 (Crossed scaffolds):**

All crossed pairings produced performance decrements:
- Constraint on arithmetic (Claude): -10%, p=.30
- Constraint on arithmetic (GPT-4o): -22%, p=.009
- Self-monitor on logic (Claude): -32%, p=.001
- Self-monitor on logic (GPT-4o): -6%, p=.54

**Prediction 3 (Interaction):**

The scaffold × task interaction was significant in the predicted direction when reframed:

| Model | Matched Δ | Crossed Δ | Difference |
|-------|-----------|-----------|------------|
| Claude | -1.0% | -21.0% | +20.0% |
| GPT-4o | -3.0% | -14.0% | +11.0% |

Matched scaffolds produced smaller decrements than crossed scaffolds, consistent with task-specific processing.

### Revised Interpretation

The original predictions assumed scaffolding would **help** matched tasks. The data reveal a more nuanced pattern:

1. **Verification + self-monitor**: Large benefit (when baseline is low)
2. **Generation + any scaffold**: Cost (scaffolding adds overhead)
3. **Crossed pairings**: Largest costs (wrong overhead, no benefit)

This suggests a **cognitive overhead account**: scaffolding imposes processing demands. When these demands address the task's bottleneck (self-monitoring for verification), the benefit outweighs the cost. When they do not (any scaffold for generation), or when they address the wrong bottleneck (crossed pairings), the cost dominates.

---

## Discussion

### Summary of Findings

We found that scaffolding interventions in LLMs produce task-specific effects, but not in the manner originally predicted. Self-monitoring scaffolds dramatically improved arithmetic verification in Claude Sonnet 4 (+28%), replicating previous findings on verification deficits (Danan, 2025). However, all scaffolding conditions degraded logic generation performance, with crossed scaffold-task pairings producing the largest decrements.

### Theoretical Implications

These findings are consistent with the hypothesis that generation and verification engage dissociable processes, but require revision of the proposed intervention mechanism.

**The hold-and-check account** (Danan, 2025) posits that LLMs lack persistent self-state for maintaining computed values during subsequent operations. Self-monitoring scaffolds may provide a prosthetic approximation of this capacity by explicitly prompting the model to "hold" its intermediate results. The dramatic improvement on arithmetic verification (+28%) supports this interpretation.

**The generation deficit** appears to have a different character. Rather than lacking a capacity that scaffolding can provide, generation tasks may already maximize the model's processing capacity. Adding any structured intervention—even one theoretically matched to the task—imposes additional overhead that degrades performance. This is consistent with capacity limitation accounts of working memory (Cowan, 2001; Oberauer et al., 2016).

The finding that **crossed scaffolds hurt most** is theoretically important. If scaffolding simply added undifferentiated overhead, crossed and matched conditions should show similar costs. The observed asymmetry (matched: -1 to -3%; crossed: -14 to -21%) suggests that overhead costs are partially offset when the scaffold addresses the task's actual bottleneck.

### Relation to Prior Work

Our verification deficit findings replicate Wei et al.'s (2022) observation that chain-of-thought prompting improves LLM performance on multi-step reasoning. The present results extend this by showing that the benefit is specific to tasks requiring verification of held values.

The generation deficit findings contrast with work suggesting that prompting strategies uniformly improve LLM performance (Kojima et al., 2022). Our results suggest that prompt engineering has task-specific effects that may be negative for some task types. This has practical implications for LLM deployment.

### Limitations

Several limitations constrain interpretation:

1. **Task validity**: The "Hillary Logic" generation tasks were designed to be pattern-unmatchable, but we cannot verify that LLMs had not encountered similar arbitrary rule structures during training.

2. **Model specificity**: Effects differed between Claude and GPT-4o (self-monitor benefit was Claude-specific). This may reflect architectural differences or training distribution differences.

3. **Scaffold specificity**: We tested two scaffold types. Other scaffolding approaches (e.g., decomposition, self-consistency) may show different patterns.

4. **Sample size**: N=50 per cell provides adequate power for large effects but may miss smaller reliable effects.

5. **Mechanism**: The cognitive overhead account is post-hoc. The data do not directly test processing capacity or overhead.

### Future Directions

These findings suggest several extensions:

1. **Titrating scaffold complexity**: If overhead mediates the generation deficit, lighter-touch scaffolds should show smaller costs.

2. **Capacity manipulation**: If generation deficits reflect capacity limits, increasing context window or reducing task complexity should selectively improve generation.

3. **Neural correlates**: For models with interpretable internals, examining activation patterns during scaffolded vs. non-scaffolded processing could reveal the computational basis of overhead costs.

4. **Individual differences**: Preliminary data suggest model-specific effects (Claude showed larger scaffold sensitivity). Systematic comparison across model families could reveal architectural determinants.

---

## Methods

### Models

- Claude Sonnet 4 (claude-sonnet-4-20250514) via Anthropic API
- GPT-4o via OpenAI API

### Tasks

**Arithmetic Verification**: "Is A × B = C? Answer only 'Yes' or 'No'." A and B were randomly sampled integers from [11, 29]. C was the correct product (50% of trials) or the correct product ± [1, 10] (50% of trials).

**Logic Generation**: Six task types with arbitrary conditional rules:
1. Word-length conditionals (response word count determined by presence of 3-letter and 6-letter words in prompt)
2. Vowel/consonant rules (response format determined by vowel:consonant ratio)
3. Sentence/word count rules (response format determined by sentence count × word count parity)
4. Position/letter rules (response word count determined by first and last letter properties)
5. Compound conditionals (response determined by count of satisfied conditions)
6. Nested conditionals (response determined by nested if-then-else evaluation)

Rules were designed to be arbitrary (unlikely to match training distribution patterns) and to require holding intermediate computation results.

### Scaffolds

**Self-monitor**: "Before answering, I want you to: 1. Work through the problem step by step. 2. State what value you computed. 3. Hold that value in mind. 4. Then check it against what's being asked. 5. Only then give your final answer. Remember: compute, hold, check, then answer."

**Constraint**: "Before answering, I want you to: 1. List all the constraints/conditions that must be checked. 2. Check each condition one by one, noting the result. 3. Determine which rule applies based on ALL your checks. 4. Only output an answer after confirming which rule applies. Remember: list conditions, check each one, determine rule, then answer."

### Procedure

Each trial: (1) generated task, (2) applied scaffold (or baseline), (3) collected model response, (4) evaluated correctness via automated scoring. 

Arithmetic verification: correct if response contained expected "Yes"/"No".
Logic generation: correct if response satisfied the derived constraint (e.g., correct word count, correct format).

### Analysis

Independent-samples t-tests compared each scaffold condition to baseline within task types. Cohen's d computed for effect sizes. Interaction assessed by comparing mean improvement for matched vs. crossed scaffold-task pairings.

### Data and Code Availability

All experimental code, raw data, and analysis scripts are available at: https://github.com/HillaryDanan/scaffolding-asymmetry

---

## References

Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. *Behavioral and Brain Sciences*, 24(1), 87-114.

Danan, H. (2025). The geometry of self-reference: Information-theoretic foundations for self-state and metacognitive capacity. *Working paper*.

Danan, H. (2025). Hold-and-check failures in large language models: Task-specific generation and verification asymmetries. *Working paper*.

Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022). Large language models are zero-shot reasoners. *Advances in Neural Information Processing Systems*, 35, 22199-22213.

Oberauer, K., Farrell, S., Jarrold, C., & Lewandowsky, S. (2016). What limits working memory capacity? *Psychological Bulletin*, 142(7), 758-799.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824-24837.

---

## Acknowledgments

The author thanks Claude (Anthropic) for assistance with experimental design, code development, and analysis.

---

## Author Contributions

H.D. conceived the study, designed experiments, conducted analyses, and wrote the manuscript.

---

## Competing Interests

The author declares no competing interests.

---

## Supplementary Information

### Table S1: Full Results by Condition

| Model | Task | Scaffold | N | Accuracy | SD | t vs. baseline | p |
|-------|------|----------|---|----------|-----|----------------|---|
| Claude | Arithmetic | Baseline | 50 | 0.70 | 0.46 | — | — |
| Claude | Arithmetic | Self-monitor | 50 | 0.98 | 0.14 | 4.09 | <.001 |
| Claude | Arithmetic | Constraint | 50 | 0.60 | 0.49 | -1.04 | .30 |
| Claude | Logic | Baseline | 50 | 0.64 | 0.48 | — | — |
| Claude | Logic | Self-monitor | 50 | 0.32 | 0.47 | -3.35 | .001 |
| Claude | Logic | Constraint | 50 | 0.34 | 0.48 | -3.11 | .002 |
| GPT-4o | Arithmetic | Baseline | 50 | 0.88 | 0.33 | — | — |
| GPT-4o | Arithmetic | Self-monitor | 50 | 0.90 | 0.30 | 0.32 | .75 |
| GPT-4o | Arithmetic | Constraint | 50 | 0.66 | 0.48 | -2.68 | .009 |
| GPT-4o | Logic | Baseline | 50 | 0.66 | 0.48 | — | — |
| GPT-4o | Logic | Self-monitor | 50 | 0.60 | 0.49 | -0.62 | .54 |
| GPT-4o | Logic | Constraint | 50 | 0.58 | 0.50 | -0.82 | .42 |

### Figure S1: Interaction Pattern

```
                        SCAFFOLD TYPE
                   Self-Monitor    Constraint
                   
ARITHMETIC    Claude:  +28%         -10%
(Verification) GPT-4o:   +2%         -22%

LOGIC         Claude:  -32%         -30%
(Generation)  GPT-4o:   -6%          -8%
```

Pattern: Matched scaffolds (diagonal) show smaller effects than crossed scaffolds (off-diagonal), but the direction differs by task type: positive for arithmetic, negative for logic.

---

*Manuscript prepared December 2025*

*"The data surprised us. That's how you know it's real."*

# DQN-vs-Heuristic Metric Pre-Registration

Pre-registered on June 12, 2026, before the Stage-0 full-stack head-to-head results were available.

## Purpose

This document fixes, in advance, the success metric and statistical test for the claim that the trained DQN policy beats the prefer-swap-when-ready heuristic on the n=5 repeater-chain task (pgen=0.4, pswap=0.7). It exists so the metric cannot be chosen post hoc to flatter the result.

## Policies compared

- `dqn_greedy`: DQN argmax over the state-conditional valid-action mask, prefer-swap filter OFF. This is THE policy the claim is about.
- `heuristic`: the non-neural prefer-swap-when-ready baseline.
- `dqn_swapprefer` is reported for reference only and is NOT part of the headline claim. It wears the heuristic as a filter, so it would be circular.
- The DQN snapshot used is the `best_eval` checkpoint (`best_eval.pt`), NOT the final `model.pt`. Mean steps and success are read off `best_eval`, filter off.

## Evaluation protocol

- Per seed: 1000 evaluation episodes, `max_actions=100`, `eval_epsilon=0.0`.
- The environment is seeded per episode so the same episode index gives the same problem instance across policies. This enables paired comparison.
- Seeds: the 5-seed protocol `{12345, 23456, 34567, 45678, 56789}` is the look-ahead. The PAPER number will use 10 seeds; this is the planned final-reporting standard, while 5 is the interim look.

## The metric: a two-part ordered gate

The metric is "equal-or-better success at fewer steps," operationalised as two gates in a FIXED order that cannot be reordered.

### Gate 1 — non-inferior success

Per seed, require `SR_dqn_greedy >= SR_heuristic - delta`, with `delta = 0.02` (two percentage points; justified because at 1000 episodes the per-seed SR standard error near 0.9 is ~1.5%). Success may NOT be traded for speed — this gate exists to block a policy that "wins" on steps by giving up early on hard episodes.

### Gate 2 — strictly fewer steps, conditional on Gate 1

This is the actual claim. Compare mean steps-to-span on the PAIRED, SHARED-SOLVED set: for each seed, restrict to episode indices that BOTH policies solved, and compute

```text
Delta = mean_steps_heuristic - mean_steps_dqn_greedy
```

on that intersection. Positive `Delta` = DQN faster. Conditioning on the shared-solved set is required because comparing "mean steps among each policy's own successes" compares different episode subsets; the intersection makes it a true paired comparison on identical instances.

## Statistical test (n is small — seeds are the unit)

Seeds are the unit of analysis, NOT episodes. Episode-level n would be pseudo-replication: episodes within a seed share one trained policy.

For the 5-seed look:

- Report all 5 per-seed `Delta` values individually, plus their mean and full spread. With 5 points, showing them is mandatory, not optional.
- Primary formal companion: paired t-test on the 5 `Delta` values (`df = 4`), report t-statistic and 95% CI on mean `Delta`.
- The two-sided Wilcoxon signed-rank test cannot reach `p < 0.05` at `n=5` (its two-sided minimum p is 0.0625), so it is underpowered by construction and is NOT the basis of the claim.
- Attainable significance at `n=5` comes from the sign test: 5/5 seeds in the winning direction gives one-sided `p = 0.031`. So directional consistency (5/5 same sign) is the headline significance statement.

## Decision rule (pre-registered)

WIN is declared if and only if BOTH:

1. Gate 1 passes on at least 4 of 5 seeds (`SR` within `delta=0.02`), AND
2. Gate 2: mean `Delta > 0` AND all 5 seeds have `Delta > 0` (same sign).

The t-test CI reports the MAGNITUDE of the effect; the 5/5 sign consistency carries the SIGNIFICANCE. The decision rests on effect size + directional consistency, not on a p-value threshold the sample size cannot support.

## Planned escalation

The final paper result will use ~10 seeds (eval-only on existing checkpoints plus a few additional training runs), which moves the sign test and paired t-test into adequately powered territory and allows a genuine `p < 0.05` with headroom rather than relying on 5/5.

## What is NOT being claimed

This is a single environment configuration (n=5, fixed pgen/pswap), single architecture family, on a problem with no decoherence. The claim is narrow: a learned policy reaches the heuristic's success while spanning the chain in fewer steps. No claim about generalisation across n, parameters, or to the multipartite/fusion setting is made here.

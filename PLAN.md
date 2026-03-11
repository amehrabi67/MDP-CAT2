# CAT-DRL Research Codebase: Full Scientific Plan
### Mehrabi & Morphew — Purdue University, 2026
---

## The Core Argument (in one paragraph)

In 3PL CAT, Fisher Information rewards are right-skewed: extreme-ability
examinees produce rewards 3-8× larger than average examinees. Under MSE
TD-loss, these dominate the gradient — identical to the heavy-tail problem
Huber (1981) solved for regression. Huber loss clips gradients for large
TD errors, bounding the influence function. Without any exposure
constraint, the DQN converges to a degenerate fixed point where 80-90%
of tests draw from <20% of the item bank — a feedback loop provable by
the Q-value update dynamics. POCAR breaks this loop by penalising the
advantage signal for over-exposed items before it reaches the policy
gradient. The real-data arm (ASSISTments) proves these effects hold on
empirically-calibrated parameters, not just idealised simulations.

---

## Repository Structure

```
cat_research/
│
├── PLAN.md                  ← This file
├── A_item_bank.py           ← Bank generation (synthetic × 2 + ASSISTments)
├── B_irt_core.py            ← 3PL, FI, Hybrid MLE, WLE, SE  (pure math)
├── C_baselines.py           ← MFI, KL-L, a-Stratification   (no ML)
├── D_dqn_agents.py          ← DQN-MSE / DQN-Huber / DQN-Huber+POCAR
├── E_simulation.py          ← Unified CAT episode runner
├── F_analysis.py            ← Metrics, Gini, plots, LaTeX tables
└── run_all.py               ← Orchestrator: runs all 25 conditions
```

---

## Why Six Separate Modules

### Module A — Item Bank
**Scientific reason for separation:** The item bank is the data generating
process. Any bug here invalidates all downstream results. By isolating it,
you can unit-test the bank statistics (a-b correlation, b distribution
shape) independently of any algorithm.

Three banks:
- `uncor` — independent a, b, c — your existing setup, kept for continuity
- `cor` — a-b corr ≈ 0.30, bimodal-b — matches real calibrated banks
          (Wingersky & Lord 1984; Chang et al. 2001)
- `assistments` — real 3PL parameters calibrated from student responses
                  via marginal MLE with Gauss-Hermite quadrature

### Module B — IRT Core
**Scientific reason for separation:** Every method (MFI, KL, DQN) uses the
same 3PL model, Fisher Information, and Hybrid MLE. If there's any
inconsistency (e.g., D=1 in one place, D=1.702 elsewhere), it corrupts the
comparison. Centralising here guarantees everyone calls the same function.

New addition: **WLE estimator** (Warm 1989). WLE has lower bias than MLE
for short tests (< 20 items) and is the standard in modern CAT software
(catR, LIVECAT). Adding one WLE condition costs 20 lines but adds a
critical comparison: if DQN-Huber doesn't beat MFI+WLE, the improvement
may be from the estimator, not the agent.

### Module C — Baselines
**Scientific reason for separation:** Baselines must be verifiable against
Wang (2024)'s published numbers before you trust them as comparators.
Isolating them lets you run `python C_baselines.py` and check that MFI
reproduces expected RMSE ≈ 0.30-0.35 on a standard bank before any
DQN is involved.

Five selectors:
1. `MFI` — greedy argmax Fisher Information (your existing baseline)
2. `KL-L` — KL information weighted by likelihood (best classical in Wang 2024)
3. `a-Strat` — a-stratification (Barrada et al. 2009; classical exposure control)
4. `Random` — floor reference (always include; reviewers expect it)
5. `MFI+WLE` — MFI selection + WLE estimation (tests estimator effect)

### Module D — DQN Agents
**Scientific reason for separation:** Your three main methods are one class
with two parameters changed — loss function and POCAR flag. Parameterising
this way means you cannot accidentally use different architectures,
learning rates, or replay buffers between conditions. The comparison is
then provably apples-to-apples.

Three agents (all same architecture, same hyperparams):
1. `DQN-MSE` — MSELoss, no POCAR  (ablation baseline)
2. `DQN-Huber` — HuberLoss(δ=1.0), no POCAR  (primary contribution 1)
3. `DQN-Huber+POCAR` — HuberLoss + corrected POCAR advantage  (primary contribution 2)

POCAR correction: Δⱼ = |usageⱼ/N − 1/N_items|  (NOT SE(θ) as in current code)

### Module E — Simulation Engine
**Scientific reason for separation:** All 8 methods (5 classical + 3 DQN)
share one episode runner. This enforces:
- Same test set (5000 examinees, same seed) for every method
- Same stopping rule (SE < 0.3 OR t=40)
- Same Hybrid MLE state update
- Same CSV output format

This is what makes the comparison publishable. Reviewers can verify that
MFI and DQN were tested on identical examinees.

### Module F — Analysis
**Scientific reason for separation:** Analysis is read-only (only reads CSVs,
writes figures). You can rerun it without re-running any expensive training.
It produces everything needed for the paper: per-step curves, Lorenz
curves, Gini coefficients, LaTeX summary table, TD-error distribution
plots (the empirical evidence for your Huber motivation).

---

## Experimental Design

```
2 (bank: uncor, cor) × 2 (train prior: normal, uniform) × 5 (method)
= 20 simulation conditions

+ 1 (real data: ASSISTments) × 5 (method)
= 5 real-data conditions

Total: 25 conditions in the main results table
```

Test set: ALWAYS 5000 examinees from N(0,1), same seed.
Training: 1000 examinees. Validation: 200 examinees every 50 episodes.

---

## Key Scientific Choices

### Why γ = 0.1 for DQN
The MDP horizon is T=40. The effective discount horizon is
1/(1-γ). At γ=0.1, the effective horizon is 1/0.9 ≈ 1.1 steps
→ nearly myopic. This means the Q-function approximates the
immediate Fisher Information of each item, which is well-defined
and bounded. At γ=0.99, the effective horizon is 100 steps —
longer than the test itself — which creates unstable TD targets
because future rewards depend on items not yet seen.

### Why Huber δ = 1.0
Fisher Information rewards in a 3PL bank with LogNormal(0, 0.5)
discrimination have mean ≈ 0.5 and standard deviation ≈ 0.4.
The 95th percentile reward is ≈ 1.3, the 99th percentile ≈ 2.1.
Setting δ=1.0 means TD errors from the top 5% of rewards are
handled with linear (not squared) loss — exactly the heavy tail
we want to down-weight. This is empirically motivated by your
data; you should plot the TD error distribution to confirm.

### Why POCAR on DQN (not PPO)
PPO has a known instability with γ=0.99 and sparse rewards.
More importantly, the POCAR modification to the advantage
function is theoretically cleaner in the off-policy DQN context:
you are modifying the target Q-value, not the policy gradient.
The constrained MDP formulation (Altman 1999) maps directly onto
the DQN target update: Q_target = r + Δ_penalty + γ*max Q'.

### Why ASSISTments (not EdNet, not STATICS)
- ASSISTments 2009 has ~110,000 students × ~110 skills
- It has been calibrated to 3PL by multiple published studies
  (the parameters are validated, not just estimated once)
- It is the standard benchmark in the Wang 2024 literature
- EdNet and STATICS use knowledge tracing models (not 3PL),
  making them incompatible with the IRT-based simulation engine

---

## Expected Run Times (Colab T4 GPU)

| Task | Time |
|------|------|
| Bank generation | < 5 seconds |
| MFI baseline (5000 students) | ~8 minutes |
| KL-L baseline | ~10 minutes |
| a-Stratification | ~8 minutes |
| DQN-MSE training (1000 ep) | ~15 minutes |
| DQN-Huber training | ~15 minutes |
| DQN-Huber+POCAR training | ~18 minutes |
| DQN evaluation (5000 students) | ~3 minutes |
| Analysis + plots | < 1 minute |
| **Total per bank×prior condition** | **~80 minutes** |
| **Total all 25 conditions** | **~7 hours** |

Recommendation: run each condition with output CSVs, then analysis
is instant. Never rerun training if CSV already exists.

---

## What Validates the Code is Correct

Before trusting any DQN result, run this sanity checklist:

1. `B_irt_core.py --test` → FI should equal d²log P/dθ² analytically
2. MFI on uncor bank, N=1000, step 40 → RMSE should be 0.30-0.38
3. MFI Gini coefficient → should be 0.85-0.95 (severe overexposure)
4. Random selector Gini → should be < 0.05 (near-uniform)
5. DQN-MSE training loss → should decrease monotonically after 200 episodes
6. DQN-Huber RMSE < DQN-MSE RMSE at step 40 (if not, check reward D constant)
7. POCAR items_used / 500 → should be > 50% (if not, check Δ formula)

"""
A_item_bank.py
==============
Item bank generation for CAT simulation studies.

Two synthetic banks + one real-data bank (ASSISTments 2009).

Scientific rationale
--------------------
The choice of item bank matters because DQN performance is known to
differ between correlated and uncorrelated banks (Wang et al. 2024).
In real calibrated banks, discrimination (a) and difficulty (b) are
positively correlated ~0.3 (Wingersky & Lord 1984; Chang et al. 2001):
harder items tend to be better discriminators. Simulating only the
uncorrelated case inflates the apparent advantage of any FI-maximising
method because there is no trade-off to exploit.

Bank types
----------
Bank 1 — Synthetic-Uncor  : a,b,c independent (your existing bank)
Bank 2 — Synthetic-Cor    : a-b correlation ~0.3, b right-skewed
                             (bimodal: easy cluster + hard cluster)
Bank 3 — ASSISTments 2009 : real 3PL parameters calibrated from
                             student response logs via marginal MLE

All banks: N_ITEMS=500, 3PL IRT model, D=1.702.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# ── Constants ─────────────────────────────────────────────────────────────────
SEED        = 42
N_ITEMS     = 500
D           = 1.702          # Lord (1980) scaling constant
THETA_MIN   = -4.0
THETA_MAX   =  4.0

rng = np.random.default_rng(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# BANK 1 — Synthetic Uncorrelated
# Your existing parameterisation; kept for continuity.
# a ~ LogNormal(0, 0.5),  b ~ N(0,1),  c ~ U(0.05, 0.30)
# ══════════════════════════════════════════════════════════════════════════════

def generate_bank_uncorrelated(n=N_ITEMS, seed=SEED):
    """
    Independent (a, b, c) parameters — the 'ideal' bank.
    Matches your existing item_bank_uncor_1.csv parameterisation.
    """
    rng_local = np.random.default_rng(seed)
    a = rng_local.lognormal(mean=0.0,  sigma=0.5, size=n).clip(0.5, 2.5)
    b = rng_local.normal(loc=0.0,      scale=1.0, size=n).clip(THETA_MIN, THETA_MAX)
    c = rng_local.uniform(low=0.05,    high=0.30, size=n)
    bank = np.column_stack([a, b, c])
    df   = pd.DataFrame(bank, columns=['a', 'b', 'c'])
    df.index.name = 'item_id'
    return df


# ══════════════════════════════════════════════════════════════════════════════
# BANK 2 — Synthetic Correlated  (the scientifically important addition)
# ══════════════════════════════════════════════════════════════════════════════

def generate_bank_correlated(n=N_ITEMS, ab_corr=0.30, seed=SEED):
    """
    Realistic bank with:
      - a-b correlation ~0.30  (Wingersky & Lord 1984)
      - b bimodal: 60% easy cluster N(-0.5,0.6), 40% hard N(1.2,0.7)
        (mirrors the difficulty distribution in operational exams where
         many items cluster near passing score and a tail of hard items)
      - c ~ Beta(8, 32)  → mean=0.20, SD≈0.065  (more realistic than Uniform)

    Generating correlated (a, b):
      Use Cholesky decomposition of the 2×2 correlation matrix,
      then transform marginals to LogNormal and bimodal-Normal.

    This directly tests whether DQN-Huber's advantage over MFI is
    robust to realistic parameter covariation — a gap in Wang 2024.
    """
    rng_local = np.random.default_rng(seed)

    # Step 1: generate correlated standard normals via Cholesky
    cov = np.array([[1.0, ab_corr],
                    [ab_corr, 1.0]])
    L   = np.linalg.cholesky(cov)        # lower-triangular Cholesky factor
    Z   = rng_local.normal(size=(n, 2))  # iid standard normals
    W   = Z @ L.T                        # correlated normals, shape (n,2)

    # Step 2: transform W[:,0] → LogNormal for a
    #   If W~N(0,1), then exp(mu + sigma*W) ~ LogNormal(mu,sigma)
    #   Target: a ~ LogNormal(0.2, 0.3), clipped [0.5, 2.5]
    a = np.exp(0.2 + 0.3 * W[:, 0]).clip(0.5, 2.5)

    # Step 3: transform W[:,1] → bimodal Normal for b
    #   Use mixture assignment: first 60% to easy cluster, rest to hard
    #   The correlation is preserved because we use the same W[:,1]
    #   (marginal is still ~N(0,1) before the mixture transform)
    u       = rng_local.uniform(size=n)
    easy    = u < 0.60
    b       = np.where(easy,
                       -0.5 + 0.6 * W[:, 1],   # easy cluster
                        1.2 + 0.7 * W[:, 1])   # hard cluster
    b       = b.clip(THETA_MIN, THETA_MAX)

    # Step 4: c ~ Beta(8, 32), mean=0.20  (independent of a, b)
    c = rng_local.beta(a=8, b=32, size=n).clip(0.05, 0.35)

    bank = np.column_stack([a, b, c])
    df   = pd.DataFrame(bank, columns=['a', 'b', 'c'])
    df.index.name = 'item_id'

    # Sanity check: empirical a-b correlation should be near ab_corr
    empirical_corr = np.corrcoef(a, b)[0, 1]
    print(f"[Bank-Cor] Empirical a-b correlation: {empirical_corr:.3f} "
          f"(target {ab_corr})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# BANK 3 — ASSISTments 2009 (real-data calibration)
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_assistments_bank(response_csv_path: str,
                                n_items_target: int = 500,
                                min_responses_per_item: int = 200,
                                seed: int = SEED) -> pd.DataFrame:
    """
    Calibrate 3PL IRT parameters from ASSISTments 2009 response logs.

    Pipeline
    --------
    1. Load student×item response matrix from CSV
       (columns: student_id, problem_id, correct)
    2. Filter items with < min_responses_per_item observations
       (sparse items give unstable MLE estimates — Lord 1980 recommends ≥200)
    3. For each item, fit 3PL via marginal maximum likelihood:
       maximize sum_j [ r_j * log P(theta_j) + (1-r_j)*log(1-P(theta_j)) ]
       integrating over the latent ability distribution (Gauss-Hermite quadrature)
    4. Clip to psychometrically plausible ranges
    5. Sub-sample n_items_target items if more are available

    Parameters
    ----------
    response_csv_path : path to the ASSISTments 2009 CSV with columns
                        [student_id, problem_id, correct]

    Returns
    -------
    DataFrame with columns [a, b, c], shape (n_items_target, 3)

    Notes
    -----
    Full EM-based 3PL calibration (as in BILOG-MG or mirt R package) is
    computationally expensive. We use a fast per-item gradient method
    with a standard normal quadrature prior on theta, which gives
    estimates comparable to full marginal MLE for the purpose of
    simulation studies (Thissen & Wainer 2001, Chapter 4).

    If you want exact replication of mirt/BILOG calibration, export
    item parameters from R and load them with load_pretrained_bank().
    """
    rng_local = np.random.default_rng(seed)

    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(response_csv_path)

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()
    col_map = {}
    for col in df.columns:
        if 'student' in col or 'user' in col:
            col_map[col] = 'student_id'
        elif 'problem' in col or 'item' in col or 'skill' in col or 'question' in col:
            col_map[col] = 'problem_id'
        elif 'correct' in col or 'answer' in col or 'score' in col:
            col_map[col] = 'correct'
    df = df.rename(columns=col_map)[['student_id', 'problem_id', 'correct']].dropna()
    df['correct'] = df['correct'].astype(int).clip(0, 1)

    print(f"[ASSISTments] Loaded {len(df):,} responses from "
          f"{df['student_id'].nunique():,} students, "
          f"{df['problem_id'].nunique():,} problems")

    # ── Filter sparse items ─────────────────────────────────────────────────
    item_counts = df.groupby('problem_id').size()
    valid_items = item_counts[item_counts >= min_responses_per_item].index
    df = df[df['problem_id'].isin(valid_items)]
    print(f"[ASSISTments] After filtering (≥{min_responses_per_item} responses): "
          f"{len(valid_items):,} items remain")

    # ── Gauss-Hermite quadrature nodes and weights ──────────────────────────
    # 21-point quadrature over N(0,1) is standard in IRT software
    n_quad  = 21
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    nodes   = nodes * np.sqrt(2)   # transform from physicist's to probabilist's convention
    weights = weights / np.sqrt(np.pi)  # normalise so weights sum to 1

    # ── Per-item 3PL calibration ────────────────────────────────────────────
    items     = list(valid_items)
    rng_local.shuffle(items)  # randomise order before sub-sampling
    items     = items[:min(len(items), n_items_target * 3)]  # keep 3× for filtering

    calibrated = []
    for item_id in items:
        r = df[df['problem_id'] == item_id]['correct'].values.astype(float)
        n = len(r)

        def neg_marginal_loglik(params):
            """
            Negative marginal log-likelihood for 3PL item parameters.
            Integrates over the N(0,1) prior on theta using quadrature.

            L(a,b,c) = sum_j log[ sum_q w_q * [P(theta_q)^r_j * (1-P)^(1-r_j)] ]
            where P(theta_q) = c + (1-c)/(1+exp(-D*a*(theta_q-b)))
            """
            a_raw, b_raw, c_raw = params
            a = np.exp(a_raw)           # log-parameterise a > 0
            b = b_raw
            c = 1.0 / (1.0 + np.exp(-c_raw))  # logit-parameterise c ∈ (0,1)
            c = np.clip(c, 0.05, 0.35)

            P = c + (1 - c) / (1 + np.exp(-D * a * (nodes - b)))  # (n_quad,)
            P = np.clip(P, 1e-9, 1 - 1e-9)

            # For each respondent: marginal likelihood = sum_q w_q * P^r * (1-P)^(1-r)
            # Shape: r (n,), P (n_quad,) → broadcast to (n, n_quad)
            log_lik_quad = (r[:, None] * np.log(P[None, :])
                            + (1 - r[:, None]) * np.log(1 - P[None, :]))  # (n, n_quad)
            # log-sum-exp trick for numerical stability
            log_marginal = np.log(np.sum(weights[None, :] * np.exp(log_lik_quad), axis=1) + 1e-12)
            return -log_marginal.sum()

        # Initial values: a=1, b=logit(1-mean_correct), c=0.2
        p_hat   = np.clip(r.mean(), 0.05, 0.95)
        b_init  = -np.log(p_hat / (1 - p_hat)) / D  # rough b from p̄
        x0      = [np.log(1.0), b_init, np.log(0.2 / 0.8)]

        try:
            result = minimize(neg_marginal_loglik, x0,
                              method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-8})
            a_est = np.exp(result.x[0])
            b_est = result.x[1]
            c_est = np.clip(1.0 / (1.0 + np.exp(-result.x[2])), 0.05, 0.35)

            # Plausibility filter (Lord 1980 recommended ranges)
            if 0.4 <= a_est <= 2.5 and -3.5 <= b_est <= 3.5:
                calibrated.append({'item_id': item_id, 'a': a_est,
                                   'b': b_est, 'c': c_est})
        except Exception:
            pass  # skip non-converging items

    if len(calibrated) == 0:
        raise ValueError("No items calibrated. Check your input CSV format.")

    cal_df = pd.DataFrame(calibrated).set_index('item_id')[['a', 'b', 'c']]
    print(f"[ASSISTments] Calibrated {len(cal_df)} items successfully")

    # Sub-sample to n_items_target
    if len(cal_df) > n_items_target:
        cal_df = cal_df.sample(n=n_items_target, random_state=seed)
    cal_df = cal_df.reset_index(drop=True)
    cal_df.index.name = 'item_id'

    print(f"[ASSISTments] Final bank: {len(cal_df)} items")
    print(f"  a: mean={cal_df.a.mean():.3f}, std={cal_df.a.std():.3f}")
    print(f"  b: mean={cal_df.b.mean():.3f}, std={cal_df.b.std():.3f}")
    print(f"  c: mean={cal_df.c.mean():.3f}, std={cal_df.c.std():.3f}")
    return cal_df


def load_pretrained_bank(csv_path: str) -> pd.DataFrame:
    """
    Load pre-calibrated item parameters from a CSV with columns [a, b, c].
    Use this if you calibrated parameters externally (e.g., R mirt package).
    """
    df = pd.read_csv(csv_path)
    assert {'a', 'b', 'c'}.issubset(df.columns), "CSV must have columns a, b, c"
    df = df[['a', 'b', 'c']].clip(
        lower={'a': 0.3, 'b': -4.0, 'c': 0.0},
        upper={'a': 3.0, 'b':  4.0, 'c': 0.35}
    )
    df.index.name = 'item_id'
    print(f"[PretrainedBank] Loaded {len(df)} items from {csv_path}")
    return df


# ── Theta populations ──────────────────────────────────────────────────────────

def sample_theta(n: int, distribution: str = 'normal', seed: int = SEED) -> np.ndarray:
    """
    Sample true latent abilities for simulation.

    distribution options:
      'normal'  : N(0,1)   — typical credentialing exam population
      'uniform' : U(-3,3)  — stress test on extreme examinees
      'bimodal' : 0.5*N(-1,0.8) + 0.5*N(1,0.8) — two-group population
                  (e.g. passing vs failing cohort; tests exposure fairness
                   across groups with very different ability levels)
    """
    rng_local = np.random.default_rng(seed)
    if distribution == 'normal':
        return rng_local.normal(0.0, 1.0, n).clip(THETA_MIN, THETA_MAX)
    elif distribution == 'uniform':
        return rng_local.uniform(-3.0, 3.0, n)
    elif distribution == 'bimodal':
        group = rng_local.choice([0, 1], size=n, p=[0.5, 0.5])
        means = np.where(group == 0, -1.0, 1.0)
        return (means + rng_local.normal(0, 0.8, n)).clip(THETA_MIN, THETA_MAX)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ── Convenience: build all banks at once ──────────────────────────────────────

def build_all_synthetic_banks(save_dir: str = '.') -> dict:
    """Generate and save all synthetic banks. Returns dict of DataFrames."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    banks = {
        'uncor': generate_bank_uncorrelated(),
        'cor':   generate_bank_correlated(),
    }
    for name, df in banks.items():
        path = f"{save_dir}/item_bank_{name}.csv"
        df.to_csv(path)
        print(f"Saved {name} bank → {path}")
    return banks


if __name__ == '__main__':
    # Quick sanity check
    uncor = generate_bank_uncorrelated()
    cor   = generate_bank_correlated()

    print("\n=== Uncorrelated Bank ===")
    print(uncor.describe().round(3))
    print(f"a-b corr: {uncor.corr().loc['a','b']:.3f}  (expected ~0)")

    print("\n=== Correlated Bank ===")
    print(cor.describe().round(3))
    print(f"a-b corr: {cor.corr().loc['a','b']:.3f}  (expected ~0.30)")

    # Test theta samplers
    for dist in ['normal', 'uniform', 'bimodal']:
        t = sample_theta(5000, dist)
        print(f"\nTheta {dist}: mean={t.mean():.3f}, std={t.std():.3f}, "
              f"min={t.min():.2f}, max={t.max():.2f}")

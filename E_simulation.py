"""
E_simulation.py
===============
Unified CAT simulation engine.

Every method — whether MFI, KL-L, a-Strat, or any DQN agent —
runs through this engine. This guarantees:

  1. Identical test population (same 5000 examinees, same seed)
  2. Identical stopping rule (SE < 0.3 OR t = 40)
  3. Identical Hybrid MLE state updates
  4. Identical CSV output format

The CSV format matches your existing records files:
    columns: [userID, step, itemID, resp, theta_est, bias]

This is what makes the comparison publishable.
A reviewer who sees "MFI and DQN were both run through
E_simulation.run_condition()" can be confident they were evaluated
on exactly the same examinees.

The engine handles both classical selectors (BaseSelector subclasses)
and DQN agents (DQNAgent) through duck typing:
    - Classical selectors: selector.select(theta_hat, used, adm_p, resp)
    - DQN agents: agent.select_test(theta_hat, used)
The engine detects which interface to call automatically.
"""

import numpy as np
import pandas as pd
import os
import time
from typing import Union

from B_irt_core import (
    P3PL, simulate_response, hybrid_mle, se_theta,
    D, THETA_MIN, THETA_MAX, SE_STOP
)


# ══════════════════════════════════════════════════════════════════════════════
# Core episode runner (single examinee)
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(selector_or_agent,
                item_bank:   np.ndarray,
                theta_true:  float,
                test_length: int   = 40,
                se_threshold: float = SE_STOP,
                record_steps: bool = True) -> dict:
    """
    Run one complete CAT session for one examinee.

    Parameters
    ----------
    selector_or_agent : BaseSelector subclass or DQNAgent
    item_bank         : (N_items, 3) — full item bank
    theta_true        : true latent ability of this examinee
    test_length       : maximum items to administer
    se_threshold      : early stopping criterion SE(θ̂) < se_threshold
    record_steps      : if True, record theta_est at every step (for plots)
                        if False, record only final step (faster)

    Returns
    -------
    dict with keys:
        steps       : list of step records (empty if record_steps=False except final)
        final_theta : final θ̂ estimate
        final_se    : final SE(θ̂)
        n_items     : number of items administered
        items_used  : list of item indices used
    """
    is_dqn = hasattr(selector_or_agent, 'select_test')

    used      = []
    adm_p     = []          # list of (3,) arrays; converted to ndarray when needed
    resp_hist = []
    theta_hat = 0.0         # initial estimate: prior mean
    steps     = []

    for t in range(test_length):
        # ── Item selection ──────────────────────────────────────────────────
        if is_dqn:
            action = selector_or_agent.select_test(theta_hat, used)
        else:
            adm_arr = np.array(adm_p) if adm_p else np.empty((0, 3))
            action  = selector_or_agent.select(
                theta_hat, used, adm_arr, resp_hist)

        # ── Simulate response ───────────────────────────────────────────────
        r_int = int(simulate_response(item_bank[action], theta_true))

        # ── Update state ────────────────────────────────────────────────────
        used.append(action)
        adm_p.append(item_bank[action])
        resp_hist.append(r_int)
        adm_arr   = np.array(adm_p)             # (t+1, 3)
        resp_arr  = np.array(resp_hist, float)  # (t+1,)
        theta_hat = hybrid_mle(adm_arr, resp_arr, D=D)
        se        = se_theta(theta_hat, adm_arr, D=D)

        # ── Record ──────────────────────────────────────────────────────────
        if record_steps:
            steps.append({
                'step':      t + 1,
                'itemID':    action + 1,           # 1-indexed for paper
                'resp':      r_int,
                'theta_est': theta_hat,
                'bias':      theta_hat - theta_true,
                'se':        se,
            })

        # ── Early stopping ──────────────────────────────────────────────────
        if se < se_threshold:
            break

    if not record_steps:
        # Only final record
        steps.append({
            'step':      len(used),
            'itemID':    used[-1] + 1,
            'resp':      resp_hist[-1],
            'theta_est': theta_hat,
            'bias':      theta_hat - theta_true,
            'se':        se_theta(theta_hat, np.array(adm_p), D=D),
        })

    return {
        'steps':       steps,
        'final_theta': theta_hat,
        'final_se':    se_theta(theta_hat, np.array(adm_p), D=D),
        'n_items':     len(used),
        'items_used':  used,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Full condition runner (N examinees → DataFrame + CSV)
# ══════════════════════════════════════════════════════════════════════════════

def run_condition(selector_or_agent,
                  item_bank:     np.ndarray,
                  theta_test:    np.ndarray,
                  condition_name: str,
                  output_dir:    str  = '.',
                  test_length:   int  = 40,
                  se_threshold:  float = SE_STOP,
                  force_rerun:   bool = False,
                  log_every:     int  = 500) -> pd.DataFrame:
    """
    Run all N test examinees and save results as CSV.

    Output file: {output_dir}/{condition_name}.csv
    If the CSV already exists and force_rerun=False, loads and returns it.
    This means expensive DQN evaluations are never re-run if you restart.

    Parameters
    ----------
    condition_name : descriptive string, becomes the CSV filename and the
                     'method' column in analysis plots.
                     Convention: "{bank}_{prior}_{method}"
                     e.g. "uncor_normal_DQN-Huber" or "cor_uniform_MFI"

    Returns
    -------
    DataFrame with columns:
        userID, step, itemID, resp, theta_est, bias, se
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{condition_name}.csv")

    # ── Load if exists ───────────────────────────────────────────────────────
    if os.path.exists(csv_path) and not force_rerun:
        df = pd.read_csv(csv_path)
        print(f"[Load] {condition_name}: {len(df):,} rows from {csv_path}")
        return df

    # ── Run simulation ───────────────────────────────────────────────────────
    N     = len(theta_test)
    records = []
    t0    = time.time()
    print(f"[Run] {condition_name}: {N} examinees ... ", end='', flush=True)

    for j, theta_true in enumerate(theta_test):
        episode = run_episode(
            selector_or_agent, item_bank, float(theta_true),
            test_length=test_length, se_threshold=se_threshold,
            record_steps=True
        )
        for rec in episode['steps']:
            records.append({
                'userID':    j + 1,
                'step':      rec['step'],
                'itemID':    rec['itemID'],
                'resp':      rec['resp'],
                'theta_est': rec['theta_est'],
                'bias':      rec['bias'],
                'se':        rec['se'],
            })

        if (j + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate    = (j + 1) / elapsed
            eta     = (N - j - 1) / rate
            print(f"\n  {j+1}/{N} done  ({rate:.0f} ex/s, ETA {eta:.0f}s)",
                  end='', flush=True)

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s → {csv_path}  ({len(df):,} rows)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Multi-condition orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_all_conditions(selectors_and_agents: dict,
                       banks: dict,
                       theta_test: np.ndarray,
                       output_dir: str = './results',
                       test_length: int = 40,
                       force_rerun: bool = False) -> dict:
    """
    Run all (bank, method) combinations and collect DataFrames.

    Parameters
    ----------
    selectors_and_agents : dict mapping method_name → selector or DQN agent
        e.g. {'MFI': MFISelector(bank), 'DQN-Huber': DQNAgent(...)}
        Note: one agent per bank, re-trained per bank. Pass pre-trained.

    banks : dict mapping bank_name → (N_items, 3) ndarray
        e.g. {'uncor': bank_uncor, 'cor': bank_cor, 'assist': bank_assist}

    theta_test : (N,) — SAME test set used for all conditions

    Returns
    -------
    results : dict mapping condition_name → DataFrame
    """
    results = {}
    for bank_name, bank in banks.items():
        for method_name, method in selectors_and_agents.items():
            cname = f"{bank_name}_{method_name}"
            df    = run_condition(
                method, bank, theta_test, cname,
                output_dir=output_dir, test_length=test_length,
                force_rerun=force_rerun
            )
            results[cname] = df

    print(f"\n[run_all_conditions] Complete: {len(results)} conditions")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Theta test set generator (fixed seed, used by everyone)
# ══════════════════════════════════════════════════════════════════════════════

def make_test_set(n: int = 5000, distribution: str = 'normal',
                  seed: int = 99) -> np.ndarray:
    """
    Generate the canonical test set.

    Seed 99 (different from training seed 42) ensures the test population
    is not the same as the training population.
    Always N(0,1) regardless of training prior — this is the real exam
    population. The training prior is a design choice; the test population
    is a fact.
    """
    rng = np.random.default_rng(seed)
    if distribution == 'normal':
        return rng.normal(0.0, 1.0, n).clip(-4, 4)
    elif distribution == 'uniform':
        return rng.uniform(-3.0, 3.0, n)
    elif distribution == 'bimodal':
        grp = rng.choice([0, 1], n, p=[0.5, 0.5])
        mu  = np.where(grp == 0, -1.0, 1.0)
        return (mu + rng.normal(0, 0.8, n)).clip(-4, 4)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ── Quick validation ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from C_baselines import MFISelector, RandomSelector

    print("Simulation engine smoke test (N=50 examinees, T=20 items)\n")
    np.random.seed(42)
    rng = np.random.default_rng(42)

    N_ITEMS = 100
    bank = np.column_stack([
        rng.lognormal(0.0, 0.5, N_ITEMS).clip(0.5, 2.5),
        rng.normal(0.0, 1.0, N_ITEMS),
        rng.uniform(0.05, 0.25, N_ITEMS)
    ])
    theta_test = make_test_set(n=50)
    output_dir = '/tmp/cat_test'

    for sel_name, sel in [('Random', RandomSelector(bank)),
                           ('MFI',    MFISelector(bank))]:
        df = run_condition(sel, bank, theta_test, sel_name,
                           output_dir=output_dir, test_length=20)
        final = df[df['step'] == df['step'].max()]
        print(f"{sel_name:8s} | RMSE={np.sqrt((final['bias']**2).mean()):.4f} "
              f"Bias={final['bias'].mean():+.4f} "
              f"Items/examinee={df.groupby('userID')['step'].max().mean():.1f}")

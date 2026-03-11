"""
run_all.py
==========
Master orchestrator for the full experiment.

Runs all 25 conditions:
  - 2 banks (uncor, cor) × 2 training priors (normal, uniform) × 5 methods
  - 1 real-data bank (assistments) × 5 methods
  = 20 simulation + 5 real-data conditions

Usage
-----
# Full run (≈7 hours on Colab T4)
python run_all.py

# Single bank, single prior (for testing)
python run_all.py --bank uncor --prior normal

# Skip training, only run analysis (if CSVs already exist)
python run_all.py --analysis_only

# Force rerun all conditions (ignore existing CSVs)
python run_all.py --force_rerun

Google Colab usage
------------------
# Mount Drive first, then:
!python run_all.py --output_dir /content/drive/MyDrive/CAT_agent/results

Checkpointing
-------------
Each condition saves a CSV immediately on completion.
If you interrupt and restart, completed conditions are skipped.
DQN weights are saved as .pt files alongside the CSVs.

This means the 7-hour run can be split across Colab sessions:
Session 1: run baselines (30 min)
Session 2: run DQN agents (3-4 hours)
Session 3: run analysis only (< 1 min)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# ── Path setup (works both in Colab and locally) ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from A_item_bank import (
    generate_bank_uncorrelated,
    generate_bank_correlated,
    calibrate_assistments_bank,
    load_pretrained_bank,
    sample_theta,
)
from B_irt_core   import D, SE_STOP
from C_baselines  import make_selector
from D_dqn_agents import make_dqn_agent
from E_simulation import run_condition, make_test_set
from F_analysis   import run_full_analysis, plot_td_error_distribution, plot_training_convergence


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

SEED            = 42
N_ITEMS         = 500
TRAIN_SIZE      = 1000       # training episodes per agent per condition
TEST_SIZE       = 5000       # examinees in test set (same for all conditions)
TEST_LENGTH     = 40         # maximum items per session
VAL_SIZE        = 200        # validation set size
VAL_INTERVAL    = 50         # validate every N episodes

# Classical baseline names (see C_baselines.make_selector)
CLASSICAL_METHODS = ['random', 'mfi', 'mfi+wle', 'kl-l', 'a-strat']

# DQN agent names (see D_dqn_agents.make_dqn_agent)
DQN_METHODS = ['dqn-mse', 'dqn-huber', 'dqn-huber+pocar']

PRIORS = ['normal', 'uniform']


# ══════════════════════════════════════════════════════════════════════════════
# Bank builders
# ══════════════════════════════════════════════════════════════════════════════

def build_banks(args) -> dict:
    """Build or load all item banks. Returns {bank_name: (N,3) ndarray}."""
    np.random.seed(SEED)
    banks = {}

    print("=" * 60)
    print("Building item banks")
    print("=" * 60)

    # Synthetic uncorrelated
    df = generate_bank_uncorrelated(n=N_ITEMS, seed=SEED)
    banks['uncor'] = df[['a', 'b', 'c']].values
    print(f"[Bank] uncor: {len(banks['uncor'])} items, "
          f"a-b corr={np.corrcoef(banks['uncor'][:,0], banks['uncor'][:,1])[0,1]:.3f}")

    # Synthetic correlated
    df = generate_bank_correlated(n=N_ITEMS, ab_corr=0.30, seed=SEED)
    banks['cor'] = df[['a', 'b', 'c']].values
    print(f"[Bank] cor:   {len(banks['cor'])} items, "
          f"a-b corr={np.corrcoef(banks['cor'][:,0], banks['cor'][:,1])[0,1]:.3f}")

    # ASSISTments (real data)
    if args.assistments_csv:
        if os.path.exists(args.assistments_csv):
            cal_path = os.path.join(args.output_dir, 'bank_assistments_calibrated.csv')
            if os.path.exists(cal_path) and not args.force_rerun:
                df = load_pretrained_bank(cal_path)
                print(f"[Bank] assistments: loaded from {cal_path}")
            else:
                df = calibrate_assistments_bank(args.assistments_csv,
                                                n_items_target=N_ITEMS)
                df.to_csv(cal_path, index=False)
                print(f"[Bank] assistments: calibrated and saved → {cal_path}")
            banks['assistments'] = df[['a', 'b', 'c']].values
        else:
            print(f"[Bank] WARNING: ASSISTments CSV not found at {args.assistments_csv}")
            print("         Skipping real-data conditions.")
    elif args.pretrained_bank_csv:
        df = load_pretrained_bank(args.pretrained_bank_csv)
        banks['assistments'] = df[['a', 'b', 'c']].values
        print(f"[Bank] assistments (pretrained): {len(banks['assistments'])} items")
    else:
        print("[Bank] No ASSISTments data provided. "
              "Pass --assistments_csv or --pretrained_bank_csv to add real-data arm.")

    return banks


# ══════════════════════════════════════════════════════════════════════════════
# Run one bank × prior × all methods
# ══════════════════════════════════════════════════════════════════════════════

def run_bank_prior_block(bank_name: str,
                          bank: np.ndarray,
                          prior: str,
                          theta_test: np.ndarray,
                          args) -> dict:
    """
    Run all 8 methods for one (bank, prior) combination.
    Returns {condition_name: DataFrame}.
    """
    print(f"\n{'='*60}")
    print(f"Block: bank={bank_name}  prior={prior}")
    print(f"{'='*60}")

    results     = {}
    ckpt_dir    = os.path.join(args.output_dir, 'checkpoints')
    results_dir = args.output_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training theta population ─────────────────────────────────────────────
    np.random.seed(SEED)
    theta_train = sample_theta(TRAIN_SIZE, prior, seed=SEED + 1)

    # ── Classical baselines ───────────────────────────────────────────────────
    for method_str in CLASSICAL_METHODS:
        if args.bank and args.bank != bank_name:
            continue
        if args.prior and args.prior != prior:
            continue

        cname    = f"{bank_name}_{prior}_{method_str.replace('+', 'plus')}"
        selector = make_selector(method_str, bank, test_length=TEST_LENGTH)
        df = run_condition(
            selector, bank, theta_test, cname,
            output_dir=results_dir,
            test_length=TEST_LENGTH,
            force_rerun=args.force_rerun,
        )
        results[cname] = df

    # ── DQN agents ────────────────────────────────────────────────────────────
    val_histories = {}

    for method_str in DQN_METHODS:
        if args.bank and args.bank != bank_name:
            continue
        if args.prior and args.prior != prior:
            continue

        # Sanitise name for file system
        safe_name = method_str.replace('+', 'plus').replace('-', '_')
        cname     = f"{bank_name}_{prior}_{method_str.replace('+', 'plus')}"
        ckpt_path = os.path.join(ckpt_dir, f"{cname}.pt")

        agent = make_dqn_agent(method_str, n_items=N_ITEMS)

        # Load checkpoint or train
        if os.path.exists(ckpt_path) and not args.force_rerun:
            print(f"\n[{agent.name}] Loading checkpoint: {ckpt_path}")
            agent.load(ckpt_path)
        else:
            print(f"\n[{agent.name}] Training (bank={bank_name}, prior={prior}) ...")
            train_result = agent.train(
                item_bank=bank,
                theta_train=theta_train,
                training_size=TRAIN_SIZE,
                test_length=TEST_LENGTH,
                validation_size=VAL_SIZE,
                validation_interval=VAL_INTERVAL,
                prior=prior,
                verbose=True,
            )
            agent.save(ckpt_path)
            val_histories[agent.name] = train_result['validation_history']

            # Save TD error distributions for Huber motivation figure
            if 'mse' in method_str:
                np.save(os.path.join(args.output_dir, f'td_errors_mse_{bank_name}_{prior}.npy'),
                        np.array(agent.training_losses))
            elif 'huber' in method_str and 'pocar' not in method_str:
                np.save(os.path.join(args.output_dir, f'td_errors_huber_{bank_name}_{prior}.npy'),
                        np.array(agent.training_losses))

        # Evaluate
        df = run_condition(
            agent, bank, theta_test, cname,
            output_dir=results_dir,
            test_length=TEST_LENGTH,
            force_rerun=args.force_rerun,
        )
        results[cname] = df

    # ── Training convergence plot ─────────────────────────────────────────────
    if val_histories:
        plot_training_convergence(
            val_histories,
            output_dir=os.path.join(args.output_dir, 'figures')
        )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TD error figure (post-hoc)
# ══════════════════════════════════════════════════════════════════════════════

def make_td_error_figures(banks_list: list, priors_list: list, output_dir: str):
    """Load saved TD error arrays and produce comparison plots."""
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    for bank_name in banks_list:
        for prior in priors_list:
            mse_path   = os.path.join(output_dir, f'td_errors_mse_{bank_name}_{prior}.npy')
            huber_path = os.path.join(output_dir, f'td_errors_huber_{bank_name}_{prior}.npy')
            if os.path.exists(mse_path) and os.path.exists(huber_path):
                td_mse   = np.load(mse_path).tolist()
                td_huber = np.load(huber_path).tolist()
                plot_td_error_distribution(
                    td_mse, td_huber, huber_delta=1.0, output_dir=fig_dir
                )


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Run all CAT-DRL experiments (Mehrabi & Morphew 2026)'
    )
    parser.add_argument('--output_dir', default='./results',
                        help='Directory for CSVs, checkpoints, figures')
    parser.add_argument('--assistments_csv', default='',
                        help='Path to ASSISTments 2009 response CSV')
    parser.add_argument('--pretrained_bank_csv', default='',
                        help='Path to pre-calibrated bank CSV (a, b, c columns)')
    parser.add_argument('--bank',  default='',
                        choices=['', 'uncor', 'cor', 'assistments'],
                        help='Run only this bank (default: all)')
    parser.add_argument('--prior', default='',
                        choices=['', 'normal', 'uniform'],
                        help='Run only this prior (default: both)')
    parser.add_argument('--analysis_only', action='store_true',
                        help='Skip all training/evaluation; only run analysis')
    parser.add_argument('--force_rerun', action='store_true',
                        help='Ignore existing CSVs and rerun everything')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # ── Analysis-only mode ───────────────────────────────────────────────────
    if args.analysis_only:
        print("[Mode] Analysis only — loading existing CSVs")
        run_full_analysis(results_dir=args.output_dir, output_dir=fig_dir)
        return

    # ── Build banks ──────────────────────────────────────────────────────────
    banks = build_banks(args)

    # ── Fixed test set (same for ALL conditions) ─────────────────────────────
    theta_test = make_test_set(n=TEST_SIZE, distribution='normal', seed=99)
    print(f"\n[Test set] N={TEST_SIZE}, θ~N(0,1), "
          f"mean={theta_test.mean():.3f}, std={theta_test.std():.3f}")
    np.save(os.path.join(args.output_dir, 'theta_test.npy'), theta_test)

    # ── Run all conditions ───────────────────────────────────────────────────
    banks_to_run = [args.bank] if args.bank else list(banks.keys())
    priors_to_run = [args.prior] if args.prior else PRIORS

    all_results = {}
    for bank_name in banks_to_run:
        if bank_name not in banks:
            print(f"[Skip] Bank '{bank_name}' not available")
            continue

        bank = banks[bank_name]

        # For ASSISTments, only run with normal prior (single condition)
        priors = priors_to_run if bank_name != 'assistments' else ['normal']

        for prior in priors:
            block = run_bank_prior_block(
                bank_name, bank, prior, theta_test, args
            )
            all_results.update(block)

    # ── TD error figures ─────────────────────────────────────────────────────
    make_td_error_figures(banks_to_run, priors_to_run,
                          output_dir=args.output_dir)

    # ── Full analysis ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Running analysis on all completed conditions")
    print(f"{'='*60}")
    run_full_analysis(results_dir=args.output_dir, output_dir=fig_dir)

    print(f"\n{'='*60}")
    print(f"All done. Results in: {args.output_dir}")
    print(f"Figures in:          {fig_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

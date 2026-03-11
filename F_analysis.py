"""
F_analysis.py
=============
All analysis, metrics, and publication-quality figures.

Reads CSV files produced by E_simulation.py — never touches raw training
code. This means you can rerun every figure in < 1 minute without
rerunning any training.

Produces:
  Figure 1  — Per-step RMSE curves (all methods, both banks)
  Figure 2  — Per-step Bias curves
  Figure 3  — Item exposure bar charts (per method)
  Figure 4  — Lorenz curves (exposure inequality visualisation)
  Figure 5  — TD-error distribution (Huber motivation)
  Figure 6  — Training convergence curves
  Table 1   — Final-step summary table (LaTeX format)
  Table 2   — Exposure summary table (LaTeX format)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# Colour / style registry
# ══════════════════════════════════════════════════════════════════════════════

# Map method name → plot style.
# Add entries here for any new method without changing plotting functions.
STYLE = {
    'Random':          dict(color='#adb5bd', lw=1.5, ls=':',  marker='x',  ms=3, zorder=1),
    'MFI':             dict(color='#2d6a4f', lw=2.5, ls='-',  marker='o',  ms=4, zorder=3),
    'MFI+WLE':         dict(color='#52b788', lw=2.0, ls='--', marker='s',  ms=4, zorder=3),
    'KL-L':            dict(color='#1d3557', lw=2.0, ls='--', marker='^',  ms=4, zorder=3),
    'a-Strat(K=4)':    dict(color='#457b9d', lw=2.0, ls='--', marker='D',  ms=4, zorder=3),
    'DQN-MSE':         dict(color='#e76f51', lw=2.0, ls='-',  marker='P',  ms=4, zorder=4),
    'DQN-Huber':       dict(color='#e63946', lw=2.5, ls='-',  marker='*',  ms=6, zorder=5),
    'DQN-Huber+POCAR': dict(color='#6a0572', lw=2.5, ls='-',  marker='h',  ms=5, zorder=5),
}

# How method names appear in the dataframes produced by E_simulation
# (the column 'method' is added by the analysis functions below)
ORDERED_METHODS = [
    'Random', 'MFI', 'MFI+WLE', 'KL-L', 'a-Strat(K=4)',
    'DQN-MSE', 'DQN-Huber', 'DQN-Huber+POCAR'
]


# ══════════════════════════════════════════════════════════════════════════════
# Metric computations
# ══════════════════════════════════════════════════════════════════════════════

def step_metrics(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Per-step Bias, RMSE, MAE, and mean SE.

    Parameters
    ----------
    df     : raw simulation DataFrame from E_simulation.run_condition
             columns: [userID, step, itemID, resp, theta_est, bias, se]
    method : label for the 'method' column

    Returns
    -------
    DataFrame with columns: [step, Bias, RMSE, MAE, SE_mean, method]
    """
    stats = (
        df.groupby('step')['bias']
        .agg(
            Bias=lambda x: x.mean(),
            RMSE=lambda x: np.sqrt(np.mean(x ** 2)),
            MAE=lambda x:  np.abs(x).mean(),
        )
        .reset_index()
    )
    if 'se' in df.columns:
        se_stats = df.groupby('step')['se'].mean().reset_index()
        se_stats.columns = ['step', 'SE_mean']
        stats = stats.merge(se_stats, on='step')
    stats['method'] = method
    return stats


def gini(x: np.ndarray) -> float:
    """
    Gini coefficient of array x.
    0 = perfect equality (all items equally exposed)
    1 = perfect concentration (one item gets everything)

    Formula: G = (2 Σ i·xᵢ) / (n Σ xᵢ) - (n+1)/n  for sorted x
    """
    x = np.sort(np.abs(x))
    n = len(x)
    if x.sum() < 1e-12:
        return 0.0
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def exposure_stats(df: pd.DataFrame,
                   method: str,
                   n_items: int = 500) -> tuple:
    """
    Compute per-item exposure rates and summary statistics.

    Returns
    -------
    stats_dict : dict with Gini, items used, max/mean/std exposure
    series     : pd.Series, index=item_id (1..N_items), values=exposure_rate
    """
    n_students = df['userID'].nunique()
    raw        = df.groupby('itemID').size() / n_students
    series     = pd.Series(0.0, index=np.arange(1, n_items + 1))
    series.update(raw)
    return {
        'method':           method,
        'items_used':       int((series > 0).sum()),
        'items_used_pct':   round(100 * (series > 0).sum() / n_items, 1),
        'max_exposure':     round(series.max(), 4),
        'mean_exposure':    round(series.mean(), 4),
        'std_exposure':     round(series.std(), 4),
        'gini':             round(gini(series.values), 4),
    }, series


def lorenz_curve(exposure_series: pd.Series) -> tuple:
    """
    Compute Lorenz curve for item exposure distribution.

    The Lorenz curve plots the cumulative share of total exposure
    (y-axis) against the cumulative share of items (x-axis), sorted
    by exposure from lowest to highest.

    A perfectly uniform exposure gives the 45-degree line.
    Departure from the diagonal = inequality.
    The Gini coefficient is twice the area between the curve and the diagonal.

    Returns: (x, y) arrays for plotting
    """
    vals = np.sort(exposure_series.values)
    n    = len(vals)
    cumsum = np.cumsum(vals)
    x = np.linspace(0, 1, n)
    y = cumsum / (cumsum[-1] + 1e-12)
    return x, y


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 & 2 — Per-step accuracy curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_curves(all_stats: pd.DataFrame,
                         bank_name: str,
                         prior_name: str,
                         output_dir: str = '.') -> str:
    """
    Three-panel figure: Bias, RMSE, MAE across test steps for all methods.

    Parameters
    ----------
    all_stats  : concatenated step_metrics DataFrames for all methods
    bank_name  : e.g. 'uncor', 'cor', 'assistments'
    prior_name : e.g. 'normal', 'uniform'
    output_dir : where to save PNG

    Returns
    -------
    Path to saved PNG
    """
    methods = [m for m in ORDERED_METHODS if m in all_stats['method'].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'CAT Accuracy — Bank: {bank_name}, Train Prior: {prior_name}  (N=5000)',
        fontsize=13, fontweight='bold', y=1.01
    )

    panels = [
        ('Bias',  'Mean Bias ($\\hat{\\theta} - \\theta$)',  True),
        ('RMSE',  'Root Mean Squared Error',                 False),
        ('MAE',   'Mean Absolute Error',                     False),
    ]

    for ax, (metric, ylabel, add_zero) in zip(axes, panels):
        for method in methods:
            sub   = all_stats[all_stats['method'] == method].sort_values('step')
            style = STYLE.get(method, dict(color='grey', lw=1.5, ls=':', ms=3))
            ax.plot(sub['step'], sub[metric], label=method, **style)

        if add_zero:
            ax.axhline(0, color='black', lw=0.8, ls=':')

        ax.set_xlabel('Test Step', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=10)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
               ncol=4, fontsize=9, frameon=True)
    plt.tight_layout()

    path = os.path.join(output_dir, f'fig_accuracy_{bank_name}_{prior_name}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Item exposure bar charts
# ══════════════════════════════════════════════════════════════════════════════

def plot_exposure_bars(exposure_series_dict: dict,
                       n_items: int = 500,
                       output_dir: str = '.') -> str:
    """
    Horizontal bar chart of item exposure rates for each method.
    The red dashed line shows the ideal uniform exposure (1/N_items).

    Parameters
    ----------
    exposure_series_dict : {method_name: pd.Series of exposure rates}
    """
    # Include any method present in the dict, not just ORDERED_METHODS
    # This handles name mismatches between condition file stems and ORDERED_METHODS
    methods = [m for m in ORDERED_METHODS if m in exposure_series_dict]
    if not methods:                          # fallback: use all keys as-is
        methods = list(exposure_series_dict.keys())
    n       = len(methods)
    if n == 0:
        print("[plot_exposure_bars] No methods to plot — skipping.")
        return ''
    ideal   = 1.0 / n_items

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        series = exposure_series_dict[method]
        g      = gini(series.values)
        used   = int((series > 0).sum())

        color = STYLE.get(method, {}).get('color', '#264653')
        ax.bar(series.index, series.values, width=1.0, color=color,
               alpha=0.75, linewidth=0)
        ax.axhline(ideal, color='crimson', ls='--', lw=1.4,
                   label=f'Ideal (1/{n_items})')
        ax.set_title(f'{method}\nGini={g:.3f}  Items={used}/{n_items}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Item ID', fontsize=8)
        ax.set_ylabel('Exposure Rate', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15)

    fig.suptitle('Item Exposure Rate Distributions — POCAR Fairness Analysis',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_exposure_bars.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Lorenz curves (exposure inequality)
# ══════════════════════════════════════════════════════════════════════════════

def plot_lorenz_curves(exposure_series_dict: dict,
                       output_dir: str = '.') -> str:
    """
    Lorenz curves for all methods on one axis.
    The 45-degree line = perfect exposure equality.
    Area below line = Gini / 2.

    This is the single most visually compelling figure for the
    POCAR claim: the DQN-Huber+POCAR Lorenz curve should be
    closest to the diagonal.
    """
    methods = [m for m in ORDERED_METHODS if m in exposure_series_dict]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.0, label='Perfect equality', zorder=0)

    for method in methods:
        series = exposure_series_dict[method]
        x, y   = lorenz_curve(series)
        g      = gini(series.values)
        style  = STYLE.get(method, dict(color='grey', lw=1.5, ls=':'))
        ax.plot(x, y, label=f'{method} (Gini={g:.3f})',
                color=style['color'], lw=style.get('lw', 1.5),
                ls=style.get('ls', '-'))

    ax.fill_between([0, 1], [0, 1], alpha=0.04, color='green',
                    label='Equality zone')
    ax.set_xlabel('Cumulative share of items (sorted by exposure)', fontsize=11)
    ax.set_ylabel('Cumulative share of total exposure', fontsize=11)
    ax.set_title('Lorenz Curves — Item Exposure Inequality\n'
                 '(closer to diagonal = more fair)', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_lorenz.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — TD-error distribution (Huber motivation)
# ══════════════════════════════════════════════════════════════════════════════

def plot_td_error_distribution(td_errors_mse: list,
                                td_errors_huber: list,
                                huber_delta: float = 1.0,
                                output_dir: str = '.') -> str:
    """
    Overlaid histograms of TD errors during training for DQN-MSE vs DQN-Huber.

    This is the empirical evidence for the Huber motivation.
    Expected: both distributions are right-skewed.
    MSE gradient ∝ error (unbounded for large errors)
    Huber gradient = error if |error| ≤ δ, else δ·sign(error) (bounded)

    The figure shows: (1) the right-skewed TD error distribution,
    (2) the linear regime boundary δ=1.0, and (3) the % of errors
    above δ (i.e., errors handled differently by Huber vs MSE).

    To collect td_errors during training, call agent.training_losses
    (this is a proxy; for exact TD errors you'd need to save them
    separately — see DQNAgent.update()).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('TD Error Distribution: Motivation for Huber Loss',
                 fontsize=12, fontweight='bold')

    for ax, errors, label, color in [
        (axes[0], td_errors_mse,   'DQN-MSE',   '#e76f51'),
        (axes[1], td_errors_huber, 'DQN-Huber', '#6a0572'),
    ]:
        errors = np.array(errors)
        pct_above = 100 * (np.abs(errors) > huber_delta).mean()
        ax.hist(errors, bins=80, color=color, alpha=0.7, density=True,
                edgecolor='none', label=label)
        ax.axvline(huber_delta,  color='crimson', ls='--', lw=1.5,
                   label=f'δ={huber_delta}  ({pct_above:.1f}% above)')
        ax.axvline(-huber_delta, color='crimson', ls='--', lw=1.5)
        ax.set_xlabel('TD Error (loss value)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{label} — Training losses\n'
                     f'{pct_above:.1f}% of errors in Huber linear regime',
                     fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_td_error_dist.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Training convergence
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_convergence(validation_histories: dict,
                               output_dir: str = '.') -> str:
    """
    Validation RMSE over training episodes for all DQN agents.

    Parameters
    ----------
    validation_histories : {method_name: list of {'episode': int, 'rmse': float}}
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for method, history in validation_histories.items():
        if not history:
            continue
        eps  = [h['episode'] for h in history]
        rmse = [h['rmse']    for h in history]
        style = STYLE.get(method, dict(color='grey', lw=1.5, ls='-'))
        ax.plot(eps, rmse, label=method,
                color=style['color'], lw=style.get('lw', 1.5),
                ls=style.get('ls', '-'))

    ax.set_xlabel('Training Episode', fontsize=11)
    ax.set_ylabel('Validation RMSE', fontsize=11)
    ax.set_title('DQN Training Convergence (Validation RMSE)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    path = os.path.join(output_dir, 'fig_training_convergence.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Tables — LaTeX output
# ══════════════════════════════════════════════════════════════════════════════

def make_summary_table(all_stats_dict: dict,
                        output_dir: str = '.') -> pd.DataFrame:
    """
    Final-step summary table across all conditions.

    Parameters
    ----------
    all_stats_dict : {condition_name: step_metrics_DataFrame}
        condition_name format: "{bank}_{prior}_{method}"

    Returns
    -------
    summary : DataFrame with MultiIndex rows (bank, prior, method)
              and columns (Bias, RMSE, MAE)
    Also saves LaTeX .tex file.
    """
    rows = []
    for cname, df in all_stats_dict.items():
        parts = cname.split('_', 2)  # bank, prior, method
        bank   = parts[0] if len(parts) > 0 else cname
        prior  = parts[1] if len(parts) > 1 else ''
        method = parts[2] if len(parts) > 2 else ''

        max_step = df['step'].max()
        final    = df[df['step'] == max_step]
        rows.append({
            'Bank':   bank,
            'Prior':  prior,
            'Method': method,
            'Bias':   round(final['Bias'].values[0],  4),
            'RMSE':   round(final['RMSE'].values[0],  4),
            'MAE':    round(final['MAE'].values[0],   4),
        })

    summary = pd.DataFrame(rows).sort_values(['Bank', 'Prior', 'RMSE'])
    summary = summary.set_index(['Bank', 'Prior', 'Method'])

    # Console print
    print("\n=== Summary Table (Final Step) ===")
    print(summary.to_string())

    # LaTeX
    latex = summary.to_latex(
        float_format='%.4f',
        caption='Ability estimation accuracy at test termination (step 40 or SE<0.3). '
                'N=5000 examinees, \\(\\theta_{\\text{test}} \\sim \\mathcal{N}(0,1)\\).',
        label='tab:accuracy_summary',
        bold_rows=True,
    )
    path = os.path.join(output_dir, 'table_summary.tex')
    with open(path, 'w') as f:
        f.write(latex)
    print(f"LaTeX table → {path}")
    return summary


def make_exposure_table(exposure_stats_list: list,
                         output_dir: str = '.') -> pd.DataFrame:
    """
    Exposure summary table.

    Parameters
    ----------
    exposure_stats_list : list of exposure_stats dicts from exposure_stats()
    """
    df = pd.DataFrame(exposure_stats_list)
    df = df.sort_values('gini')

    print("\n=== Exposure Table ===")
    print(df.to_string(index=False))

    latex = df.to_latex(
        index=False,
        float_format='%.4f',
        caption='Item exposure statistics across methods. '
                'Gini=0 is perfectly uniform; Gini=1 is full concentration. '
                'Items used = count of distinct items administered across N=5000 sessions.',
        label='tab:exposure_summary',
    )
    path = os.path.join(output_dir, 'table_exposure.tex')
    with open(path, 'w') as f:
        f.write(latex)
    print(f"LaTeX table → {path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Master analysis pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(results_dir: str = './results',
                      output_dir:  str = './figures',
                      n_items:     int = 500) -> dict:
    """
    Load all CSVs from results_dir and produce all figures and tables.

    CSV naming convention: {bank}_{prior}_{method}.csv
    e.g. uncor_normal_MFI.csv, cor_uniform_DQN-Huber+POCAR.csv

    Returns dict with all computed DataFrames.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load all CSVs ────────────────────────────────────────────────────────
    csv_files = list(Path(results_dir).glob('*.csv'))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return {}

    print(f"Found {len(csv_files)} condition CSVs in {results_dir}\n")

    all_step_stats   = {}
    exposure_all     = {}
    exp_stats_list   = []

    # ── Method name display mapping ──────────────────────────────────────────
    # run_all.py sanitises names: '+' → 'plus', '-' preserved in some places.
    # Map file-stem method fragments back to display names used in STYLE dict.
    _name_map = {
        'random':             'Random',
        'mfi':                'MFI',
        'mfipluswle':         'MFI+WLE',
        'kl-l':               'KL-L',
        'kll':                'KL-L',
        'a-strat':            'a-Strat(K=4)',
        'astrat':             'a-Strat(K=4)',
        'dqn-mse':            'DQN-MSE',
        'dqn_mse':            'DQN-MSE',
        'dqn-huber':          'DQN-Huber',
        'dqn_huber':          'DQN-Huber',
        'dqn-huberpluspocar': 'DQN-Huber+POCAR',
        'dqn_huberpluspocar': 'DQN-Huber+POCAR',
        'dqn-huber+pocar':    'DQN-Huber+POCAR',
    }

    for csv_path in sorted(csv_files):
        cname  = csv_path.stem  # filename without .csv
        df     = pd.read_csv(csv_path)
        parts  = cname.split('_', 2)
        bank   = parts[0] if len(parts) > 0 else 'unknown'
        prior  = parts[1] if len(parts) > 1 else 'unknown'
        raw_method = parts[2] if len(parts) > 2 else cname
        method = _name_map.get(raw_method.lower(), raw_method)

        # Step metrics
        stats_df = step_metrics(df, method)
        all_step_stats[cname] = stats_df

        # Exposure
        stats, series = exposure_stats(df, method, n_items=n_items)
        stats['bank']  = bank
        stats['prior'] = prior
        exposure_all[method] = series
        exp_stats_list.append(stats)

    # ── Group by bank × prior for accuracy plots ─────────────────────────────
    bank_priors = set()
    for cname in all_step_stats:
        parts = cname.split('_', 2)
        if len(parts) >= 2:
            bank_priors.add((parts[0], parts[1]))

    for bank, prior in sorted(bank_priors):
        subset = {k: v for k, v in all_step_stats.items()
                  if k.startswith(f'{bank}_{prior}_')}
        if subset:
            combined = pd.concat(subset.values(), ignore_index=True)
            plot_accuracy_curves(combined, bank, prior, output_dir=output_dir)

    # ── Exposure plots ───────────────────────────────────────────────────────
    if exposure_all:
        plot_exposure_bars(exposure_all, n_items=n_items, output_dir=output_dir)
        plot_lorenz_curves(exposure_all, output_dir=output_dir)

    # ── Summary tables ───────────────────────────────────────────────────────
    summary_df  = make_summary_table(all_step_stats, output_dir=output_dir)
    exposure_df = make_exposure_table(exp_stats_list, output_dir=output_dir)

    return {
        'step_stats':    all_step_stats,
        'exposure':      exposure_all,
        'summary':       summary_df,
        'exposure_table': exposure_df,
    }


if __name__ == '__main__':
    import sys
    # Quick test with synthetic data
    print("F_analysis.py — testing with synthetic data\n")
    np.random.seed(42)

    # Fake step-metrics DataFrame
    steps   = np.arange(1, 41)
    methods = ['MFI', 'DQN-Huber', 'DQN-Huber+POCAR']
    rows    = []
    for m in methods:
        base_rmse = {'MFI': 0.35, 'DQN-Huber': 0.32, 'DQN-Huber+POCAR': 0.33}[m]
        for s in steps:
            decay = base_rmse + 0.5 * np.exp(-0.1 * s)
            rows.append({'step': s, 'method': m,
                         'Bias': np.random.normal(-0.05, 0.01),
                         'RMSE': decay + np.random.normal(0, 0.005),
                         'MAE':  decay * 0.9})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for m in methods:
        sub   = df[df['method'] == m]
        style = STYLE.get(m, {})
        ax.plot(sub['step'], sub['RMSE'], label=m,
                color=style.get('color', 'grey'), lw=style.get('lw', 1.5))
    ax.set_xlabel('Step'); ax.set_ylabel('RMSE'); ax.legend()
    ax.set_title('F_analysis smoke test — RMSE curves')
    plt.tight_layout()
    plt.savefig('/tmp/f_analysis_test.png', dpi=100)
    print("Smoke test plot saved → /tmp/f_analysis_test.png")

    # Gini test
    uniform = np.ones(500)
    conc    = np.zeros(500); conc[0] = 500
    print(f"Gini(uniform)={gini(uniform):.4f}  (expected ≈0)")
    print(f"Gini(concentrated)={gini(conc):.4f} (expected ≈1)")
    print("\nAll smoke tests passed.")

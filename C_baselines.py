"""
C_baselines.py
==============
Classical CAT item selection baselines.

All five selectors inherit from BaseSelector so the simulation engine
(E_simulation.py) can call them with a unified interface:
    selector.select(theta_hat, used_items, administered_params, responses)

Scientific references
---------------------
MFI      : Lord (1980); Birnbaum (1968)
KL-L     : Chang & Ying (1996), Biometrika 83(2):447-452
a-Strat  : Barrada et al. (2009), Applied Psychological Measurement 33(7)
           Chang, Qian & Ying (2001), Psychometrika 66(2):249-270
WLE      : Warm (1989), Psychometrika 54(3):427-450
Random   : standard floor reference; inclusion expected by reviewers

These baselines must reproduce Wang (2024)'s numbers before being
trusted as comparators for the DQN agents. Specifically:
  - MFI on a 500-item LogNormal/Normal/Uniform bank, 5000 examinees,
    test length 40: RMSE should be 0.30–0.38.
  - KL-L should match or slightly beat MFI on RMSE.
  - MFI Gini coefficient should be 0.85–0.95 (severe overexposure).
"""

import numpy as np
from abc import ABC, abstractmethod
from B_irt_core import (
    P3PL, fisher_info, kl_information,
    hybrid_mle, wle, se_theta,
    D, THETA_MIN, THETA_MAX
)


# ══════════════════════════════════════════════════════════════════════════════
# Base class — uniform interface for all selectors
# ══════════════════════════════════════════════════════════════════════════════

class BaseSelector(ABC):
    """Abstract base for all item selectors."""

    def __init__(self, item_bank: np.ndarray, name: str):
        """
        Parameters
        ----------
        item_bank : (N_items, 3) — [a, b, c] for all items in bank
        name      : human-readable label for plots and tables
        """
        self.bank  = np.atleast_2d(item_bank)
        self.name  = name
        self.n     = len(self.bank)

    @abstractmethod
    def select(self,
               theta_hat:   float,
               used_items:  list,
               adm_params:  np.ndarray,
               responses:   np.ndarray) -> int:
        """
        Select the next item index.

        Parameters
        ----------
        theta_hat   : current ability estimate θ̂_t
        used_items  : list of already-administered item indices
        adm_params  : (t, 3) params of items administered so far
        responses   : (t,) responses so far

        Returns
        -------
        item_idx : int — index into item_bank
        """
        ...

    def _mask(self, scores: np.ndarray, used_items: list) -> np.ndarray:
        """Set scores of used items to -inf so they cannot be re-selected."""
        scores = scores.copy()
        if used_items:
            scores[used_items] = -np.inf
        return scores

    def reset(self):
        """Called at the start of each new examinee. Override if stateful."""
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 1. Random Selector  (floor reference)
# ══════════════════════════════════════════════════════════════════════════════

class RandomSelector(BaseSelector):
    """
    Uniformly random item selection — the floor reference.
    Expected RMSE ≈ 0.55–0.65 at step 40 on standard banks.
    Expected Gini ≈ 0.01–0.05 (near-perfect exposure uniformity).
    Always include: reviewers expect it to confirm MFI > Random.
    """

    def __init__(self, item_bank: np.ndarray):
        super().__init__(item_bank, 'Random')

    def select(self, theta_hat, used_items, adm_params, responses) -> int:
        available = [i for i in range(self.n) if i not in set(used_items)]
        return int(np.random.choice(available))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Maximum Fisher Information (MFI)
# ══════════════════════════════════════════════════════════════════════════════

class MFISelector(BaseSelector):
    """
    Greedy Maximum Fisher Information selector.
    Selects item j* = argmax_j I_j(θ̂_t).

    This is the de facto standard in IRT-based CAT and the primary
    comparator in Wang (2024). It maximises immediate precision but
    ignores long-term test efficiency and causes severe item overexposure
    (Gini ≈ 0.85-0.95 in typical 500-item banks).
    """

    def __init__(self, item_bank: np.ndarray, estimator: str = 'mle'):
        """
        Parameters
        ----------
        estimator : 'mle' (default) or 'wle'
                    Controls theta estimation (not item selection).
                    MFI+WLE is a separate condition in the paper.
        """
        name = 'MFI' if estimator == 'mle' else 'MFI+WLE'
        super().__init__(item_bank, name)
        self.estimator = estimator

    def select(self, theta_hat, used_items, adm_params, responses) -> int:
        fi     = fisher_info(self.bank, theta_hat, D=D)  # (N_items,)
        fi     = self._mask(fi, used_items)
        return int(fi.argmax())

    def estimate_theta(self, adm_params, responses) -> float:
        """Call the correct estimator based on self.estimator."""
        if self.estimator == 'wle':
            return wle(adm_params, responses, D=D)
        return hybrid_mle(adm_params, responses, D=D)


# ══════════════════════════════════════════════════════════════════════════════
# 3. KL Information Weighted by Likelihood (KL-L)
# ══════════════════════════════════════════════════════════════════════════════

class KLLSelector(BaseSelector):
    """
    KL information weighted by the posterior likelihood (KL-L).

    Selector (Chang & Ying 1996):
        j* = argmax_j  KL_j(θ̂) * L(θ̂ | responses)

    The likelihood weight makes the selector less susceptible to
    early-stage estimation error than pure KL information.

    In Wang (2024), KL-L consistently outperformed MFI on RMSE,
    making it the strongest classical baseline for this paper.
    Any method that claims to beat MFI must also beat KL-L to be
    credible.

    Implementation note:
    We use the second-order Taylor approximation for KL:
        KL_j(θ̂) ≈ ½ I_j(θ̂) δ²
    with δ=0.5 (Chang & Ying 1996 recommendation).
    The likelihood weight is computed at the current θ̂ estimate.
    """

    def __init__(self, item_bank: np.ndarray, delta: float = 0.5):
        super().__init__(item_bank, 'KL-L')
        self.delta = delta

    def select(self, theta_hat, used_items, adm_params, responses) -> int:
        # KL information at current θ̂
        kl = kl_information(self.bank, theta_hat, delta=self.delta, D=D)  # (N_items,)

        # Likelihood weight: L(θ̂ | responses) = Π P(Xⱼ|θ̂)
        if len(responses) > 0 and len(adm_params) > 0:
            P    = P3PL(np.atleast_2d(adm_params), theta_hat, D=D)   # (t,)
            r    = np.asarray(responses, dtype=float)
            logL = np.sum(r * np.log(P + 1e-12) + (1 - r) * np.log(1 - P + 1e-12))
            # Use exp(logL/t) to avoid numerical underflow on long tests
            t    = max(len(responses), 1)
            L_weight = np.exp(logL / t)
        else:
            L_weight = 1.0

        scores = kl * L_weight
        scores = self._mask(scores, used_items)
        return int(scores.argmax())


# ══════════════════════════════════════════════════════════════════════════════
# 4. a-Stratification (Chang, Qian & Ying 2001; Barrada et al. 2009)
# ══════════════════════════════════════════════════════════════════════════════

class AStratSelector(BaseSelector):
    """
    a-Stratification with b-blocking (Chang et al. 2001).

    Mechanism:
    1. Divide items into K strata by discrimination (a-parameter),
       low-a items in stratum 1, high-a items in stratum K.
    2. In the first T/K steps, select from stratum 1 (low discriminators)
       using MFI within that stratum.
    3. In steps T/K+1 to 2T/K, select from stratum 2, etc.

    Rationale:
    Early in testing, θ̂ is imprecise. Using high-discrimination items
    early causes strong selection bias toward items near the prior θ̂=0.
    Low-discrimination items are more robust to estimation error at early
    steps, allowing the θ̂ to stabilise before the high-a items are used.

    This is the most-cited exposure control method in psychometrics
    (>1000 citations). Including it gives the paper a direct comparison
    to a well-understood classical fairness mechanism.

    Parameters
    ----------
    n_strata    : number of a-strata (default 4, matching Barrada 2009)
    test_length : total test length (used to compute stratum boundaries)
    """

    def __init__(self,
                 item_bank: np.ndarray,
                 n_strata: int = 4,
                 test_length: int = 40):
        super().__init__(item_bank, f'a-Strat(K={n_strata})')
        self.n_strata    = n_strata
        self.test_length = test_length

        # Pre-sort items into strata by discrimination
        a_vals          = self.bank[:, 0]
        strata_bounds   = np.percentile(a_vals, np.linspace(0, 100, n_strata + 1))
        self.strata_idx = []
        for k in range(n_strata):
            lo = strata_bounds[k]
            hi = strata_bounds[k + 1]
            if k < n_strata - 1:
                stratum = np.where((a_vals >= lo) & (a_vals < hi))[0]
            else:
                stratum = np.where((a_vals >= lo) & (a_vals <= hi))[0]
            self.strata_idx.append(stratum.tolist())

        # Steps per stratum
        self.steps_per_stratum = max(1, test_length // n_strata)

        print(f"[a-Strat] Strata item counts: "
              f"{[len(s) for s in self.strata_idx]}")
        print(f"[a-Strat] Steps per stratum: {self.steps_per_stratum}")

    def _current_stratum(self, step: int) -> int:
        """Which stratum to select from at the current step (0-indexed)."""
        k = min(step // self.steps_per_stratum, self.n_strata - 1)
        return k

    def select(self, theta_hat, used_items, adm_params, responses) -> int:
        step    = len(used_items)
        k       = self._current_stratum(step)
        stratum = self.strata_idx[k]

        # Within-stratum MFI
        fi_all = np.full(self.n, -np.inf)
        fi_all[stratum] = fisher_info(self.bank[stratum], theta_hat, D=D)
        fi_all          = self._mask(fi_all, used_items)

        # Fallback: if all stratum items exhausted, use global MFI
        if fi_all.max() == -np.inf:
            fi_all                = fisher_info(self.bank, theta_hat, D=D)
            fi_all                = self._mask(fi_all, used_items)

        return int(fi_all.argmax())


# ══════════════════════════════════════════════════════════════════════════════
# Factory function — create selector by name
# ══════════════════════════════════════════════════════════════════════════════

def make_selector(name: str,
                  item_bank: np.ndarray,
                  test_length: int = 40) -> BaseSelector:
    """
    Factory: create a selector by name string.

    Names: 'random', 'mfi', 'mfi+wle', 'kl-l', 'a-strat'
    """
    n = name.lower().strip()
    if n == 'random':
        return RandomSelector(item_bank)
    elif n == 'mfi':
        return MFISelector(item_bank, estimator='mle')
    elif n in ('mfi+wle', 'mfi_wle'):
        return MFISelector(item_bank, estimator='wle')
    elif n in ('kl-l', 'kll', 'kl_l'):
        return KLLSelector(item_bank)
    elif n in ('a-strat', 'astrat', 'a_strat'):
        return AStratSelector(item_bank, test_length=test_length)
    else:
        raise ValueError(f"Unknown selector '{name}'. Choose from: "
                         "random, mfi, mfi+wle, kl-l, a-strat")


# ── Quick validation ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    Run a mini-simulation (N=100 examinees, 40 items) on a toy bank
    to verify each selector produces sensible RMSE and exposure patterns.
    Expected:
        Random RMSE ≈ 0.55-0.70 (floor)
        MFI    RMSE ≈ 0.30-0.40 (best single-step)
        KL-L   RMSE ≈ 0.28-0.38 (should match or beat MFI)
        a-Strat RMSE ≈ 0.32-0.42 (slightly worse than MFI, better exposure)
    """
    import sys
    sys.path.insert(0, '.')
    from B_irt_core import simulate_response

    np.random.seed(42)
    N_ITEMS = 200
    N_TEST  = 100
    T       = 40

    # Small synthetic bank
    rng  = np.random.default_rng(42)
    bank = np.column_stack([
        rng.lognormal(0.0, 0.5, N_ITEMS).clip(0.5, 2.5),
        rng.normal(0.0, 1.0, N_ITEMS),
        rng.uniform(0.05, 0.25, N_ITEMS)
    ])
    theta_test = rng.normal(0, 1, N_TEST)

    selectors = {
        'Random': RandomSelector(bank),
        'MFI':    MFISelector(bank),
        'KL-L':   KLLSelector(bank),
        'a-Strat': AStratSelector(bank, test_length=T),
    }

    print(f"\nMini-validation: N={N_TEST} examinees, T={T} items, "
          f"bank={N_ITEMS} items\n{'='*60}")

    for sel_name, selector in selectors.items():
        biases = []
        item_counts = np.zeros(N_ITEMS)

        for j in range(N_TEST):
            used      = []
            adm_p     = np.empty((0, 3))
            resp_hist = []
            theta_hat = 0.0

            for t in range(T):
                idx      = selector.select(theta_hat, used, adm_p, resp_hist)
                r        = int(simulate_response(bank[idx], theta_test[j]))
                used.append(idx)
                adm_p    = np.vstack([adm_p, bank[idx]]) if len(adm_p) else bank[idx:idx+1]
                resp_hist.append(r)
                item_counts[idx] += 1
                theta_hat = hybrid_mle(adm_p, np.array(resp_hist))

            biases.append(theta_hat - theta_test[j])

        biases = np.array(biases)
        used_count = int((item_counts > 0).sum())
        max_exp    = item_counts.max() / N_TEST
        # Gini
        x     = np.sort(item_counts)
        n     = len(x)
        gini  = (2 * np.sum(np.arange(1, n+1) * x) / (n * x.sum() + 1e-12) - (n+1)/n)

        print(f"{sel_name:12s} | RMSE={np.sqrt(np.mean(biases**2)):.4f} "
              f"Bias={biases.mean():+.4f} | "
              f"Items used: {used_count}/{N_ITEMS} "
              f"Max exp: {max_exp:.2f}  Gini: {gini:.3f}")

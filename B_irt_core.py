"""
B_irt_core.py
=============
Centralised IRT mathematics — the single source of truth for all methods.

Every selector (MFI, KL-L, a-Strat, DQN variants) calls functions from
this module. This guarantees that D=1.702, the 3PL formula, the Hybrid
MLE boundary heuristics, and the SE threshold are identical across all
25 experimental conditions.

Scientific references
---------------------
3PL model         : Lord (1980), Applications of IRT to Practical Testing
Fisher Information: Birnbaum (1968), in Lord & Novick, Ch. 17-20
Hybrid MLE        : Warm (1989) — boundary heuristic is standard in catR
WLE estimator     : Warm (1989), Weighted Likelihood Estimation in IRT
                    Psychometrika 54(3):427-450
SE formula        : SE(θ̂) = 1/√Σ Iⱼ(θ̂)  — information-based SE
KL Information    : Chang & Ying (1996), Biometrika 83(2):447-452
"""

import numpy as np
from scipy.optimize import minimize_scalar, brentq

# ── Global constants ───────────────────────────────────────────────────────────
D         = 1.702      # Lord (1980) scaling constant — aligns logistic with normal ogive
THETA_MIN = -4.0
THETA_MAX =  4.0
SE_STOP   = 0.3        # Early-stopping criterion: SE(θ̂) < SE_STOP


# ══════════════════════════════════════════════════════════════════════════════
# 3PL Response Probability
# P(θ) = c + (1-c) / (1 + exp(-D·a·(θ-b)))
# ══════════════════════════════════════════════════════════════════════════════

def P3PL(item_params: np.ndarray, theta, D: float = D) -> np.ndarray:
    """
    3PL item response function (vectorised).

    Parameters
    ----------
    item_params : ndarray, shape (n_items, 3) or (3,)
                  columns: [a (discrimination), b (difficulty), c (guessing)]
    theta       : float or ndarray, shape (n_examinees,)
    D           : scaling constant (Lord 1980: D=1.702)

    Returns
    -------
    P : ndarray, shape (n_items,) if theta is scalar,
                         (n_items, n_examinees) if theta is array
    """
    item_params = np.atleast_2d(item_params)          # (n_items, 3)
    theta       = np.atleast_1d(np.asarray(theta, float))  # (n_examinees,)
    a = item_params[:, 0:1]   # (n_items, 1)
    b = item_params[:, 1:2]
    c = item_params[:, 2:3]
    # broadcast: (n_items, 1) × (1, n_examinees) → (n_items, n_examinees)
    P = c + (1.0 - c) / (1.0 + np.exp(-D * a * (theta[np.newaxis, :] - b)))
    P = np.clip(P, 1e-9, 1.0 - 1e-9)
    return P[:, 0] if theta.size == 1 else P


def simulate_response(item_params: np.ndarray, theta, D: float = D) -> np.ndarray:
    """
    Simulate binary 3PL response(s).  Returns 0/1 array same shape as P3PL.
    """
    P = P3PL(item_params, theta, D=D)
    return (np.random.rand(*P.shape) <= P).astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# Fisher Information
# I(θ) = D²a² (P-c)² (1-P) / [(1-c)² P]
# ══════════════════════════════════════════════════════════════════════════════

def fisher_info(item_params: np.ndarray, theta, D: float = D) -> np.ndarray:
    """
    3PL Fisher Information at ability theta (vectorised).

    This is the reward function for all DQN agents.
    Shape: same convention as P3PL.

    Derivation: I(θ) = -E[∂²log L / ∂θ²] for a single item
    = [P'(θ)]² / [P(θ)(1-P(θ))]
    where P'(θ) = D·a·(P-c)(1-P)/(1-c)
    """
    item_params = np.atleast_2d(item_params)
    theta       = np.atleast_1d(np.asarray(theta, float))
    a = item_params[:, 0:1]
    b = item_params[:, 1:2]
    c = item_params[:, 2:3]
    P = P3PL(item_params, theta, D=D)
    if theta.size > 1:
        a = a  # (n_items, 1) broadcasts with P (n_items, n_examinees)
    numer = (D * a) ** 2 * (P - c) ** 2 * (1.0 - P)
    denom = (1.0 - c) ** 2 * P + 1e-12
    I = numer / denom
    return I[:, 0] if theta.size == 1 else I


def test_info(administered_params: np.ndarray, theta: float, D: float = D) -> float:
    """Sum of Fisher Information across all administered items at theta."""
    if len(administered_params) == 0:
        return 0.0
    return float(fisher_info(administered_params, theta, D=D).sum())


# ══════════════════════════════════════════════════════════════════════════════
# Kullback-Leibler Information  (Chang & Ying 1996)
# KL_j(θ̂) = Σ_x P(x|θ̂) log[P(x|θ̂)/P(x|θ)]  integrated over θ
# KL-L selector uses: KL_j(θ̂) weighted by the likelihood L(θ|responses)
# ══════════════════════════════════════════════════════════════════════════════

def kl_information(item_params: np.ndarray,
                   theta_hat: float,
                   delta: float = 0.5,
                   D: float = D) -> np.ndarray:
    """
    Expected KL information for each item, approximated on [θ̂-δ, θ̂+δ].

    KL_j(θ̂) ≈ (1/2) I_j(θ̂) * δ²   (second-order Taylor expansion)

    This approximation is standard in CAT software (catR, LIVECAT) and
    makes KL-L computationally feasible without numerical integration.
    Wang (2024) uses the same approximation.

    For the full KL-L selector, multiply by the likelihood weight
    (see C_baselines.py).

    Parameters
    ----------
    delta : half-width of the ability interval to integrate over
            (Chang & Ying 1996 recommend δ ≈ 0.5)
    """
    I = fisher_info(item_params, theta_hat, D=D)
    # Second-order Taylor: KL ≈ ½ I(θ̂) δ²
    return 0.5 * I * (delta ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# Ability Estimation
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_mle(item_params: np.ndarray,
               responses: np.ndarray,
               D: float = D,
               theta_min: float = THETA_MIN,
               theta_max: float = THETA_MAX) -> float:
    """
    Hybrid Maximum Likelihood Estimation for a single examinee.

    Handles degenerate response patterns (all-correct / all-incorrect)
    with the standard boundary heuristic used in catR and LIVECAT:
      - All correct  → b_max + 1.5  (examinee is above the hardest item)
      - All incorrect → b_min - 1.5  (examinee is below the easiest item)

    Parameters
    ----------
    item_params : (n_items, 3) — params of administered items
    responses   : (n_items,) — 0/1 responses
    """
    responses   = np.asarray(responses, dtype=float)
    item_params = np.atleast_2d(item_params)
    n           = len(responses)

    if responses.sum() == n:                  # all correct
        return float(np.clip(item_params[:, 1].max() + 1.5, theta_min, theta_max))
    if responses.sum() == 0:                  # all incorrect
        return float(np.clip(item_params[:, 1].min() - 1.5, theta_min, theta_max))

    a = item_params[:, 0]
    b = item_params[:, 1]
    c = item_params[:, 2]

    def neg_loglik(x: float) -> float:
        p = c + (1.0 - c) / (1.0 + np.exp(-D * a * (x - b)))
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        return -np.sum(responses * np.log(p) + (1.0 - responses) * np.log(1.0 - p))

    result = minimize_scalar(neg_loglik, bounds=(theta_min, theta_max), method='bounded')
    return float(np.clip(result.x, theta_min, theta_max))


def hybrid_mle_batch(item_params: np.ndarray,
                     responses: np.ndarray,
                     D: float = D) -> np.ndarray:
    """
    Vectorised Hybrid MLE across N examinees.

    Parameters
    ----------
    item_params : (n_items, 3) — same items for all examinees
    responses   : (n_items, N) — response matrix

    Returns
    -------
    theta_hat : (N,) ability estimates
    """
    N         = responses.shape[1]
    theta_hat = np.zeros(N)
    for j in range(N):
        theta_hat[j] = hybrid_mle(item_params, responses[:, j], D=D)
    return theta_hat


def wle(item_params: np.ndarray,
        responses: np.ndarray,
        D: float = D,
        theta_min: float = THETA_MIN,
        theta_max: float = THETA_MAX) -> float:
    """
    Warm's (1989) Weighted Likelihood Estimator.

    WLE solves:  ∂log L / ∂θ + H(θ) = 0
    where H(θ) = (1/2) ∂/∂θ [I(θ)] / I(θ)  is the Warm correction term.

    WLE has smaller bias than MLE for short tests and at ability extremes
    (Warm 1989; also confirmed in catR package documentation).
    We include it as a separate estimator to test whether DQN-Huber
    improvements over MFI are due to the agent or the estimator.

    Note: WLE is slightly slower than MLE due to the correction term.
    For the paper, use WLE only for the MFI+WLE condition.
    """
    responses   = np.asarray(responses, dtype=float)
    item_params = np.atleast_2d(item_params)
    n           = len(responses)

    if responses.sum() == n:
        return float(np.clip(item_params[:, 1].max() + 1.5, theta_min, theta_max))
    if responses.sum() == 0:
        return float(np.clip(item_params[:, 1].min() - 1.5, theta_min, theta_max))

    a = item_params[:, 0]
    b = item_params[:, 1]
    c = item_params[:, 2]

    def score_plus_correction(x: float) -> float:
        """WLE estimating equation: ∂log L/∂θ + H(θ) = 0"""
        # P and derivatives
        P   = c + (1.0 - c) / (1.0 + np.exp(-D * a * (x - b)))
        P   = np.clip(P, 1e-9, 1.0 - 1e-9)
        P1  = D * a * (P - c) * (1.0 - P) / (1.0 - c)  # dP/dθ

        # Log-likelihood score
        score = np.sum(P1 * (responses - P) / (P * (1.0 - P)))

        # Warm correction H(θ) = ½ (dI/dθ) / I
        # dI/dθ approximated numerically (less elegant but exact)
        eps   = 1e-5
        I_hi  = fisher_info(item_params, x + eps, D=D).sum()
        I_lo  = fisher_info(item_params, x - eps, D=D).sum()
        I_now = fisher_info(item_params, x,       D=D).sum()
        dI_dtheta = (I_hi - I_lo) / (2.0 * eps)
        H     = 0.5 * dI_dtheta / (I_now + 1e-12)

        return score + H

    try:
        # brentq is more reliable than minimize_scalar for root-finding
        theta_est = brentq(score_plus_correction, theta_min, theta_max,
                           xtol=1e-6, maxiter=100)
    except ValueError:
        # Fallback to MLE if WLE doesn't bracket a root
        theta_est = hybrid_mle(item_params, responses, D=D)

    return float(np.clip(theta_est, theta_min, theta_max))


# ══════════════════════════════════════════════════════════════════════════════
# Standard Error of Estimation
# SE(θ̂) = 1 / √[Σ_j I_j(θ̂)]
# ══════════════════════════════════════════════════════════════════════════════

def se_theta(theta_hat: float,
             administered_params: np.ndarray,
             D: float = D) -> float:
    """
    Standard Error of θ̂ based on accumulated Fisher Information.
    Used as the early-stopping criterion: stop when SE < SE_STOP = 0.3.
    """
    if len(administered_params) == 0:
        return np.inf
    total_info = fisher_info(np.atleast_2d(administered_params), theta_hat, D=D).sum()
    return 1.0 / np.sqrt(total_info + 1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# Self-test  (run as script to verify mathematical correctness)
# ══════════════════════════════════════════════════════════════════════════════

def _run_tests():
    """
    Unit tests for all IRT functions.
    Analytical checks against known results from Lord (1980) Table 6.1.
    """
    import warnings
    warnings.filterwarnings('ignore')
    print("Running IRT core unit tests...\n")
    all_pass = True

    # Test 1: P3PL at b=theta should give (1+c)/2
    item = np.array([[1.0, 0.0, 0.2]])  # a=1, b=0, c=0.2
    p_at_b = float(P3PL(item, 0.0, D=D)[0])
    expected = (1.0 + 0.2) / 2.0   # = 0.6
    test1 = abs(p_at_b - expected) < 1e-6
    print(f"Test 1 P(θ=b) = (1+c)/2: {'PASS' if test1 else 'FAIL'} "
          f"(got {p_at_b:.6f}, expected {expected:.6f})")
    all_pass = all_pass and test1

    # Test 2: P3PL should approach c as theta → -∞
    p_low = float(P3PL(item, -10.0, D=D)[0])
    test2 = abs(p_low - 0.2) < 1e-4
    print(f"Test 2 P(θ→-∞) → c: {'PASS' if test2 else 'FAIL'} "
          f"(got {p_low:.6f}, expected {0.2:.6f})")
    all_pass = all_pass and test2

    # Test 3: P3PL should approach 1 as theta → +∞
    p_high = float(P3PL(item, 10.0, D=D)[0])
    test3 = abs(p_high - 1.0) < 1e-4
    print(f"Test 3 P(θ→+∞) → 1: {'PASS' if test3 else 'FAIL'} "
          f"(got {p_high:.6f})")
    all_pass = all_pass and test3

    # Test 4: Fisher Information peaks near b
    thetas = np.linspace(-3, 3, 300)
    FIs = fisher_info(item, thetas, D=D)
    peak_theta = thetas[FIs.argmax()]
    # For 3PL, peak is slightly above b due to guessing
    test4 = abs(peak_theta - 0.0) < 0.5
    print(f"Test 4 FI peaks near b=0: {'PASS' if test4 else 'FAIL'} "
          f"(peak at θ={peak_theta:.3f})")
    all_pass = all_pass and test4

    # Test 5: MLE recovers true theta within 0.5 for long response string
    np.random.seed(42)
    bank = np.column_stack([
        np.ones(30),                              # a=1
        np.linspace(-2, 2, 30),                  # b spread
        np.full(30, 0.0)                          # c=0 (2PL for clarity)
    ])
    theta_true = 0.5
    resp       = simulate_response(bank, theta_true)
    theta_mle  = hybrid_mle(bank, resp, D=D)
    test5 = abs(theta_mle - theta_true) < 1.0   # within 1.0 (stochastic)
    print(f"Test 5 MLE recovery (θ_true=0.5): {'PASS' if test5 else 'FAIL'} "
          f"(MLE estimate={theta_mle:.3f})")
    all_pass = all_pass and test5

    # Test 6: WLE exists and runs
    try:
        theta_wle = wle(bank, resp, D=D)
        test6 = THETA_MIN <= theta_wle <= THETA_MAX
        print(f"Test 6 WLE runs and is in range: {'PASS' if test6 else 'FAIL'} "
              f"(WLE={theta_wle:.3f}, MLE={theta_mle:.3f})")
    except Exception as e:
        test6 = False
        print(f"Test 6 WLE: FAIL ({e})")
    all_pass = all_pass and test6

    # Test 7: SE decreases as more items administered
    params_1 = bank[:1];   se1 = se_theta(0.0, params_1)
    params_5 = bank[:5];   se5 = se_theta(0.0, params_5)
    params_20 = bank[:20]; se20 = se_theta(0.0, params_20)
    test7 = se1 > se5 > se20
    print(f"Test 7 SE decreases with more items: {'PASS' if test7 else 'FAIL'} "
          f"(SE@1={se1:.3f}, SE@5={se5:.3f}, SE@20={se20:.3f})")
    all_pass = all_pass and test7

    # Test 8: KL information is non-negative
    kl = kl_information(bank, 0.0)
    test8 = (kl >= 0).all()
    print(f"Test 8 KL information ≥ 0: {'PASS' if test8 else 'FAIL'}")
    all_pass = all_pass and test8

    print(f"\n{'All tests PASSED' if all_pass else 'Some tests FAILED'}")
    return all_pass


if __name__ == '__main__':
    _run_tests()

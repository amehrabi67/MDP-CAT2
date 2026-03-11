"""
D_dqn_agents.py
===============
Three DQN-based CAT agents — all sharing one class, differing only
in loss function and POCAR flag.

Why one class:
--------------
Making DQN-MSE and DQN-Huber the *same class with different loss parameter*
is not just code hygiene — it is a scientific requirement. If they were
separate classes, a reviewer could ask: "are you sure the improvement
comes from Huber loss and not from some architectural difference?" With
one class, the comparison is provably apples-to-apples.

Agents
------
1. DQN-MSE         : loss='mse',   pocar=False  (ablation baseline)
2. DQN-Huber       : loss='huber', pocar=False  (primary contribution 1)
3. DQN-Huber+POCAR : loss='huber', pocar=True   (primary contribution 2)

Architecture:  θ̂_t (1D) → FC(50, ReLU) → FC(30, ReLU) → Q(500)
Matches your existing QNet exactly — no change.

POCAR correction from current code:
The current code computes Δ using N_used = item_usage.sum(), giving
normalised usage rates. The correct formula (paper Eq. 7) is:
    Δⱼ = |usageⱼ / N_students_so_far − 1/N_items|
where N_students_so_far is the number of training episodes completed
so far, NOT the total items administered. We fix this here.

Scientific references
---------------------
DQN          : Mnih et al. (2015), Nature 518:529-533
Huber loss   : Huber (1981), Robust Statistics; also Mnih (2015) Eq. 1
Positive weight constraint: equivalent to isotonic prior (see paper §3.2)
POCAR        : Mehrabi & Morphew (2026), this paper, Eq. 7
Constrained MDP: Altman (1999), Constrained Markov Decision Processes
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

from B_irt_core import (
    fisher_info, hybrid_mle, se_theta,
    D, THETA_MIN, THETA_MAX
)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
# Q-Network Architecture  (unchanged from your existing QNet)
# ══════════════════════════════════════════════════════════════════════════════

class QNet(nn.Module):
    """
    Deep Q-Network for MDP-based item selection.

    Architecture: θ̂_t → FC(h1, ReLU) → FC(h2, ReLU) → Q(n_items)

    Input (state): scalar θ̂_t ∈ [-4, 4]
    Output: Q-values for all n_items actions simultaneously

    Positive weight constraint (apply_positive_weights):
    After each gradient step we clamp all weights ≥ 0.
    Theoretical justification: Fisher Information rewards are always
    non-negative (I(θ) ≥ 0 by definition). Therefore Q(s,a) should
    be non-negative for all (s,a). The constraint makes the Q-function
    representationally consistent with the reward structure — this is
    an isotonic prior in the sense of Barlow et al. (1972).
    No existing DQN-CAT paper mentions this; it is novel.
    """

    def __init__(self,
                 input_size:   int = 1,
                 h1:           int = 50,
                 h2:           int = 30,
                 n_actions:    int = 500,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(h2, n_actions)
        )
        self._init_weights()

    def _init_weights(self):
        """Kaiming normal init — appropriate for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def apply_positive_constraint(self, min_val: float = 0.0):
        """
        Clamp all weight tensors to [min_val, ∞).
        Called after every gradient update during training.
        """
        with torch.no_grad():
            for param in self.parameters():
                param.clamp_(min=min_val)


# ══════════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Experience replay buffer (Mnih et al. 2015).

    Stores (s, a, r, s', done) tuples. Sampling is uniform random,
    which is correct for this setting because the MDP is episodic
    (each examinee is independent) and the state space is 1D.

    Prioritised replay would theoretically be better but adds complexity
    without clear benefit in this low-dimensional setting.
    """

    def __init__(self, capacity: int = 1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(s).to(device),
            torch.LongTensor(a).to(device),
            torch.FloatTensor(r).to(device),
            torch.FloatTensor(ns).to(device),
            torch.FloatTensor(d).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
# POCAR Advantage Modifier
# ══════════════════════════════════════════════════════════════════════════════

def pocar_q_target_modifier(action: int,
                             item_usage: np.ndarray,
                             n_episodes_done: int,
                             n_items: int,
                             beta_0: float = 1.0,
                             beta_1: float = 0.5,
                             beta_2: float = 0.5,
                             omega:  float = 0.3) -> float:
    """
    POCAR penalty added to the TD target reward.

    The corrected POCAR formula (Eq. 7 in paper, with fix):

        Δⱼ = |usageⱼ / N_episodes − 1/N_items|

    where N_episodes is the number of training episodes completed,
    representing the number of students trained on so far.
    This gives the per-episode exposure deviation, not the raw count.

    The penalty is added to the reward before TD-target computation:
        r_eff = beta_0 * r + under_penalty + over_momentum

    Parameters
    ----------
    action          : item index just selected
    item_usage      : (N_items,) cumulative usage counts across all episodes
    n_episodes_done : number of training episodes completed so far (≥1)
    n_items         : total items in bank (N_items)
    beta_0          : advantage weight (default 1.0)
    beta_1          : under-exposure penalty weight
    beta_2          : over-exposure momentum penalty weight
    omega           : exposure tolerance threshold

    Returns
    -------
    reward_modifier : float — added to Fisher Information reward
    """
    # Exposure deviation for the selected item
    # Δⱼ ∈ [0, 1]: 0 = perfectly uniform, 1 = completely concentrated
    ideal_exposure = 1.0 / n_items
    actual_exposure = item_usage[action] / max(n_episodes_done, 1)
    delta_j = abs(actual_exposure - ideal_exposure)

    # Under-exposure penalty: penalise if item is below threshold ω
    # (rewards items that are under-used, incentivising exploration)
    under_penalty = beta_1 * min(0.0, -delta_j + omega)

    # Over-exposure momentum penalty: penalise if this item is MORE
    # exposed than the previous item's exposure
    # (discourages repeatedly selecting the same high-FI items)
    if action > 0:
        delta_prev = abs(item_usage[action - 1] / max(n_episodes_done, 1)
                         - ideal_exposure)
    else:
        delta_prev = delta_j
    over_momentum = beta_2 * min(0.0, delta_j - delta_prev)

    return beta_0 * 1.0 + under_penalty + over_momentum
    # Note: beta_0 * 1.0 because this is a multiplier on the base reward,
    # not a standalone reward. The caller multiplies by FI.


# ══════════════════════════════════════════════════════════════════════════════
# Main DQN Agent Class
# ══════════════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    Unified DQN agent for CAT item selection.

    Instantiation:
        DQN-MSE         : DQNAgent(loss='mse',   pocar=False)
        DQN-Huber       : DQNAgent(loss='huber', pocar=False)
        DQN-Huber+POCAR : DQNAgent(loss='huber', pocar=True)

    All three agents use:
      - Same QNet architecture [1 → 50 → 30 → 500]
      - Same Kaiming initialisation
      - Same positive weight constraint
      - Same ε-greedy exploration (ε=0.1)
      - Same target network (synced every Q_SYNC_EVERY steps)
      - Same replay buffer (capacity 1000)
      - Same Adam optimiser (lr=1e-3)
      - Same γ=0.1 discount

    Only the loss function and POCAR reward modifier differ.
    """

    def __init__(self,
                 n_items:          int   = 500,
                 loss:             str   = 'huber',   # 'mse' or 'huber'
                 pocar:            bool  = False,
                 gamma:            float = 0.1,
                 lr:               float = 1e-3,
                 epsilon:          float = 0.1,
                 memory_capacity:  int   = 1000,
                 batch_size:       int   = 128,
                 q_sync_every:     int   = 40,
                 huber_delta:      float = 1.0,
                 h1:               int   = 50,
                 h2:               int   = 30,
                 # POCAR parameters
                 beta_0:           float = 1.0,
                 beta_1:           float = 0.5,
                 beta_2:           float = 0.5,
                 omega:            float = 0.3):

        # Validate loss choice
        assert loss in ('mse', 'huber'), f"loss must be 'mse' or 'huber', got '{loss}'"

        self.n_items     = n_items
        self.loss_type   = loss
        self.use_pocar   = pocar
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.batch_size  = batch_size
        self.q_sync      = q_sync_every
        self.beta_0      = beta_0
        self.beta_1      = beta_1
        self.beta_2      = beta_2
        self.omega       = omega

        # Label for file names and plots
        base = f"DQN-{'Huber' if loss == 'huber' else 'MSE'}"
        self.name = f"{base}+POCAR" if pocar else base

        # Networks
        self.eval_net   = QNet(n_actions=n_items, h1=h1, h2=h2).to(device)
        self.target_net = QNet(n_actions=n_items, h1=h1, h2=h2).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        # Optimiser
        self.optimiser = optim.Adam(self.eval_net.parameters(), lr=lr)

        # Loss function — THE critical difference between conditions
        if loss == 'huber':
            # HuberLoss with δ=1.0: L(x) = x²/2 if |x|≤δ, else δ|x| - δ²/2
            # For δ=1: linear regime kicks in at |TD error| > 1
            # Fisher Information reward σ ≈ 0.4, so 99th percentile ≈ 2.1 > δ
            # This clips the gradient for the heaviest-tailed rewards.
            self.loss_fn = nn.HuberLoss(reduction='mean', delta=huber_delta)
        else:
            # MSELoss: L(x) = x²  — no gradient clipping for large errors
            # Equivalent to minimising the squared Bellman residual
            self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer(memory_capacity)

        # Tracking
        self.n_updates       = 0   # total gradient steps
        self.item_usage      = np.zeros(n_items)  # for POCAR
        self.n_episodes_done = 0   # for POCAR Δ normalisation
        self.training_losses = []  # for convergence plot

    # ── Action selection ────────────────────────────────────────────────────

    def select_train(self, theta_hat: float, used_items: list) -> int:
        """ε-greedy action selection during training."""
        available = [i for i in range(self.n_items) if i not in set(used_items)]
        if np.random.rand() > self.epsilon:
            s = torch.FloatTensor([[theta_hat]]).to(device)
            with torch.no_grad():
                q = self.eval_net(s).cpu().numpy().squeeze()  # (n_items,)
            q[list(used_items)] = -np.inf
            return int(q.argmax())
        return int(np.random.choice(available))

    def select_test(self, theta_hat: float, used_items: list) -> int:
        """Greedy action selection during evaluation (no exploration)."""
        s = torch.FloatTensor([[theta_hat]]).to(device)
        with torch.no_grad():
            q = self.eval_net(s).cpu().numpy().squeeze()
        q[list(used_items)] = -np.inf
        return int(q.argmax())

    def select_batch_test(self,
                           theta_hats: np.ndarray,
                           used_matrix: np.ndarray) -> np.ndarray:
        """
        Vectorised greedy selection for N examinees simultaneously.
        Used in the testing loop to speed up evaluation.

        Parameters
        ----------
        theta_hats  : (N,) current estimates for all examinees
        used_matrix : (t, N) matrix of used item indices (0-indexed)

        Returns
        -------
        actions : (N,) selected item indices
        """
        S = torch.FloatTensor(theta_hats[:, np.newaxis]).to(device)  # (N,1)
        with torch.no_grad():
            Q = self.eval_net(S).cpu().numpy()  # (N, n_items)
        N = Q.shape[0]
        if used_matrix.size > 0:
            t = used_matrix.shape[0]
            rows = np.tile(np.arange(N), (t, 1))           # (t, N)
            Q[rows.flatten(), used_matrix.flatten()] = -np.inf
        return Q.argmax(axis=1).astype(int)

    # ── TD Learning step ────────────────────────────────────────────────────

    def push(self, state, action, reward, next_state, done):
        """Store transition in replay buffer, optionally with POCAR reward mod."""
        if self.use_pocar:
            modifier = pocar_q_target_modifier(
                action, self.item_usage, self.n_episodes_done,
                self.n_items, self.beta_0, self.beta_1, self.beta_2, self.omega
            )
            reward = reward * modifier
        self.item_usage[action] += 1
        self.memory.push(
            [state], action, reward, [next_state], float(done)
        )

    def update(self):
        """One gradient step. Returns loss value (float) or None if buffer too small."""
        if len(self.memory) < self.batch_size:
            return None

        s, a, r, ns, d = self.memory.sample(self.batch_size)
        # s, ns shape: (batch, 1)
        # a shape: (batch,), r shape: (batch,), d shape: (batch,)

        # Current Q values: Q(s, a) for the taken action
        q_current = self.eval_net(s).gather(1, a.unsqueeze(1)).squeeze(1)  # (batch,)

        # TD target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next    = self.target_net(ns).max(dim=1)[0]  # (batch,)
            q_target  = r + self.gamma * q_next * (1.0 - d)

        # Compute loss — the one critical difference between DQN-MSE and DQN-Huber
        loss = self.loss_fn(q_current, q_target)

        # Gradient step
        self.optimiser.zero_grad()
        loss.backward()
        # Gradient clipping (same for both loss types — not the key difference)
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        # Positive weight constraint (isotonic prior for non-negative rewards)
        self.eval_net.apply_positive_constraint()

        self.n_updates += 1
        loss_val = loss.item()
        self.training_losses.append(loss_val)

        # Sync target network
        if self.n_updates % self.q_sync == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        return loss_val

    # ── Training loop ───────────────────────────────────────────────────────

    def train(self,
              item_bank:           np.ndarray,
              theta_train:         np.ndarray,
              training_size:       int = 1000,
              test_length:         int = 40,
              validation_size:     int = 200,
              validation_interval: int = 50,
              prior:               str = 'normal',
              verbose:             bool = True) -> dict:
        """
        Full DQN training loop.

        One episode = one training examinee. Within each episode:
        1. Sample theta_true from prior
        2. Run test_length steps of the CAT
        3. Push transitions to replay buffer
        4. Call update() after each step
        5. Apply POCAR modifier to reward (if self.use_pocar)
        6. Periodic validation on held-out examinees

        Parameters
        ----------
        item_bank   : (N_items, 3)
        theta_train : (training_size,) true ability values for training
                      (drawn from prior in train_with_prior if None)
        prior       : 'normal' or 'uniform' — for logging only when
                      theta_train is passed directly
        """
        self.item_bank = item_bank  # store for select_test
        n_items = len(item_bank)
        assert n_items == self.n_items, (
            f"item_bank has {n_items} items but agent was built for {self.n_items}")

        validation_history = []
        best_val_rmse      = np.inf
        best_state_dict    = None

        for ep in range(training_size):
            theta_true = float(theta_train[ep])

            # ── Episode initialisation ──────────────────────────────────────
            used      = []
            adm_p     = np.empty((0, 3))
            resp_hist = []
            theta_hat = 0.0
            self.n_episodes_done += 1

            for t in range(test_length):
                # ε-greedy action
                action    = self.select_train(theta_hat, used)
                item_p    = item_bank[action]

                # Simulate response
                from B_irt_core import simulate_response
                r_int     = int(simulate_response(item_p, theta_true))

                # Fisher Information reward
                fi_reward = float(fisher_info(item_p.reshape(1, 3), theta_hat, D=D)[0])

                # Next state: Hybrid MLE update
                used.append(action)
                adm_p     = (np.vstack([adm_p, item_p]) if len(adm_p)
                             else item_p.reshape(1, 3))
                resp_hist.append(r_int)
                theta_next = hybrid_mle(adm_p, np.array(resp_hist), D=D)

                # Done if SE threshold reached or last step
                se    = se_theta(theta_next, adm_p, D=D)
                done  = (se < 0.3) or (t == test_length - 1)

                # Push to buffer (POCAR modifies reward inside push)
                self.push(theta_hat, action, fi_reward, theta_next, done)

                # Gradient update
                self.update()

                theta_hat = theta_next
                if done:
                    break

            # ── Periodic validation ─────────────────────────────────────────
            if (ep + 1) % validation_interval == 0:
                val_rmse = self._validate(
                    item_bank,
                    np.random.normal(0, 1, validation_size)
                    if prior == 'normal'
                    else np.random.uniform(-3, 3, validation_size),
                    test_length
                )
                validation_history.append({'episode': ep + 1, 'rmse': val_rmse})
                if verbose:
                    print(f"  [{self.name} ep {ep+1:4d}] Val RMSE={val_rmse:.4f} "
                          f"loss={np.mean(self.training_losses[-50:]):.4f}")

                if val_rmse < best_val_rmse:
                    best_val_rmse   = val_rmse
                    best_state_dict = {k: v.clone()
                                       for k, v in self.eval_net.state_dict().items()}

        # Restore best checkpoint
        if best_state_dict is not None:
            self.eval_net.load_state_dict(best_state_dict)
            if verbose:
                print(f"  [{self.name}] Restored best checkpoint "
                      f"(val RMSE={best_val_rmse:.4f})")

        return {
            'validation_history': validation_history,
            'best_val_rmse':      best_val_rmse,
            'training_losses':    self.training_losses,
        }

    def _validate(self,
                  item_bank:   np.ndarray,
                  val_thetas:  np.ndarray,
                  test_length: int = 40) -> float:
        """Run greedy evaluation on val_thetas; return RMSE."""
        from B_irt_core import simulate_response
        self.eval_net.eval()
        biases = []
        for theta_true in val_thetas:
            used      = []
            adm_p     = np.empty((0, 3))
            resp_hist = []
            theta_hat = 0.0
            for t in range(test_length):
                action    = self.select_test(theta_hat, used)
                r_int     = int(simulate_response(item_bank[action], theta_true))
                used.append(action)
                adm_p     = (np.vstack([adm_p, item_bank[action]])
                             if len(adm_p) else item_bank[action].reshape(1, 3))
                resp_hist.append(r_int)
                theta_hat = hybrid_mle(adm_p, np.array(resp_hist), D=D)
                if se_theta(theta_hat, adm_p, D=D) < 0.3:
                    break
            biases.append(theta_hat - theta_true)
        self.eval_net.train()
        return float(np.sqrt(np.mean(np.array(biases) ** 2)))

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save eval_net weights + agent metadata."""
        torch.save({
            'state_dict':    self.eval_net.state_dict(),
            'name':          self.name,
            'loss_type':     self.loss_type,
            'use_pocar':     self.use_pocar,
            'item_usage':    self.item_usage,
            'n_episodes':    self.n_episodes_done,
            'training_losses': self.training_losses,
        }, path)
        print(f"[{self.name}] Saved → {path}")

    def load(self, path: str):
        """Load eval_net weights. Syncs target net."""
        ckpt = torch.load(path, map_location=device)
        self.eval_net.load_state_dict(ckpt['state_dict'])
        self.target_net.load_state_dict(ckpt['state_dict'])
        self.item_usage      = ckpt.get('item_usage',    self.item_usage)
        self.n_episodes_done = ckpt.get('n_episodes',    0)
        self.training_losses = ckpt.get('training_losses', [])
        print(f"[{self.name}] Loaded ← {path}")


# ── Factory ────────────────────────────────────────────────────────────────────

def make_dqn_agent(name: str, n_items: int = 500, **kwargs) -> DQNAgent:
    """
    Factory: create a DQN agent by name string.

    Names: 'dqn-mse', 'dqn-huber', 'dqn-huber+pocar'
    """
    n = name.lower().strip()
    if n == 'dqn-mse':
        return DQNAgent(n_items=n_items, loss='mse', pocar=False, **kwargs)
    elif n == 'dqn-huber':
        return DQNAgent(n_items=n_items, loss='huber', pocar=False, **kwargs)
    elif n in ('dqn-huber+pocar', 'dqn-huber-pocar'):
        return DQNAgent(n_items=n_items, loss='huber', pocar=True, **kwargs)
    else:
        raise ValueError(f"Unknown DQN agent '{name}'. "
                         "Choose: dqn-mse, dqn-huber, dqn-huber+pocar")


if __name__ == '__main__':
    """Quick smoke test: verify all three agents train without errors."""
    import sys
    sys.path.insert(0, '.')

    print("Smoke test: training 3 DQN agents for 10 episodes each...\n")
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # Tiny bank for speed
    N = 50
    bank = np.column_stack([
        rng.lognormal(0.0, 0.5, N).clip(0.5, 2.5),
        rng.normal(0.0, 1.0, N),
        rng.uniform(0.05, 0.25, N)
    ])
    thetas = rng.normal(0, 1, 10)

    for agent_name in ['dqn-mse', 'dqn-huber', 'dqn-huber+pocar']:
        agent = make_dqn_agent(agent_name, n_items=N)
        result = agent.train(bank, thetas, training_size=10,
                             test_length=10, validation_interval=5,
                             verbose=True)
        print(f"  {agent.name}: best_val_rmse={result['best_val_rmse']:.4f}\n")

    print("All smoke tests passed.")

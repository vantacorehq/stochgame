"""
stochgame - stochastic games & game-theory simulation for autonomous agents.

One file. One dependency: numpy.

    python stochgame.py demo       # tour of everything
    python stochgame.py selftest   # sanity checks

Contents
  1. Matrix-game solvers : exact zero-sum LP (simplex), bimatrix Nash (support enumeration)
  2. StochasticGame      : N players, S states, P[s,a0..aN-1,s'], R[s,a0..aN-1,player]
  3. Planning            : Shapley value iteration (zero-sum stochastic games)
  4. Agents              : Random, Q-learning, Minimax-Q, Regret Matching, TFT, Grim, Pavlov ...
  5. Simulation          : play(), tournament(), replicator dynamics
  6. Game library        : PD, stag hunt, chicken, RPS, matching pennies, repeated games,
                           commons (tragedy of the commons), random games
"""
from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass

import numpy as np

__version__ = "0.1.0"

# --------------------------------------------------------------------------- #
# 1. Matrix-game solvers
# --------------------------------------------------------------------------- #


def solve_zero_sum(M):
    """Value and optimal mixed strategies of the zero-sum matrix game M
    (row player maximises). Exact, via a tableau simplex with Bland's rule.
    Returns (value, x_row, y_col)."""
    M = np.asarray(M, float)
    m, n = M.shape
    shift = M.min() - 1.0
    A = M - shift  # all entries >= 1  ->  value > 0
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = A
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = 1.0
    T[m, :n] = -1.0
    basis = list(range(n, n + m))
    for _ in range(5000):
        neg = np.flatnonzero(T[m, :-1] < -1e-12)
        if neg.size == 0:
            break
        c = neg[0]
        col = T[:m, c]
        pos = col > 1e-12
        if not pos.any():
            break
        ratio = np.full(m, np.inf)
        ratio[pos] = T[:m, -1][pos] / col[pos]
        cand = np.flatnonzero(ratio <= ratio.min() + 1e-12)
        r = min(cand, key=lambda i: basis[i])
        T[r] /= T[r, c]
        f = T[:, c].copy()
        f[r] = 0.0
        T -= np.outer(f, T[r])
        basis[r] = c
    s = T[m, -1]
    y = np.zeros(n + m)
    for i, b in enumerate(basis):
        y[b] = T[i, -1]
    y = y[:n] / s
    x = T[m, n:n + m] / s
    return 1.0 / s + shift, x, y


def nash_gap(A, B, x, y):
    """Sum of both players' incentives to deviate (0 at a Nash equilibrium)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    return float((A @ y).max() - x @ A @ y + (x @ B).max() - x @ B @ y)


def _indifferent(Msub):
    k = Msub.shape[1]
    K = np.zeros((k + 1, k + 1))
    K[:k, :k] = Msub
    K[:k, k] = -1.0
    K[k, :k] = 1.0
    rhs = np.zeros(k + 1)
    rhs[k] = 1.0
    try:
        return np.linalg.solve(K, rhs)[:k]
    except np.linalg.LinAlgError:
        return None


def nash_equilibria(A, B, tol=1e-7):
    """All Nash equilibria of a (small) bimatrix game by support enumeration
    (non-degenerate games). Returns a list of (x, y)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    m, n = A.shape
    out = []
    for k in range(1, min(m, n) + 1):
        for I in itertools.combinations(range(m), k):
            for J in itertools.combinations(range(n), k):
                y = _indifferent(A[np.ix_(I, J)])
                x = _indifferent(B[np.ix_(I, J)].T)
                if x is None or y is None or (x < -tol).any() or (y < -tol).any():
                    continue
                X, Y = np.zeros(m), np.zeros(n)
                X[list(I)], Y[list(J)] = np.clip(x, 0, None), np.clip(y, 0, None)
                if nash_gap(A, B, X, Y) < tol and not any(
                        np.allclose(X, a, atol=1e-6) and np.allclose(Y, b, atol=1e-6) for a, b in out):
                    out.append((X, Y))
    return out


# --------------------------------------------------------------------------- #
# 2. Stochastic game
# --------------------------------------------------------------------------- #


class StochasticGame:
    """N-player stochastic game.
       P: (S, A_0, ..., A_{N-1}, S)   transition probabilities
       R: (S, A_0, ..., A_{N-1}, N)   per-player rewards
    """

    def __init__(self, P, R, gamma=0.95, start=None, terminal=None, name="game"):
        self.P, self.R = np.asarray(P, float), np.asarray(R, float)
        self.N = self.R.ndim - 2
        self.S = self.P.shape[0]
        self.nA = tuple(self.P.shape[1:1 + self.N])
        if self.P.shape != (self.S, *self.nA, self.S) or self.R.shape != (self.S, *self.nA, self.N):
            raise ValueError("inconsistent P / R shapes")
        if not np.allclose(self.P.sum(-1), 1.0):
            raise ValueError("rows of P must sum to 1")
        self.gamma, self.name = gamma, name
        self.start = np.eye(self.S)[0] if start is None else np.asarray(start, float) / np.sum(start)
        self.terminal = np.zeros(self.S, bool) if terminal is None else np.asarray(terminal, bool)
        self._cdf = np.cumsum(self.P, -1)
        self.state = 0

    def reset(self, rng):
        self.state = int(rng.choice(self.S, p=self.start))
        return self.state

    def step(self, acts, rng):
        """-> (next_state, reward_vector, terminal)"""
        idx = (self.state, *acts)
        s2 = min(int(np.searchsorted(self._cdf[idx], rng.random(), side="right")), self.S - 1)
        self.state = s2
        return s2, self.R[idx], bool(self.terminal[s2])

    def stage(self, s):
        """Stage-game payoff matrices (A, B) of a 2-player game in state s."""
        return self.R[s, ..., 0], self.R[s, ..., 1]


# --------------------------------------------------------------------------- #
# 3. Planning: Shapley value iteration (2-player zero-sum)
# --------------------------------------------------------------------------- #


def shapley(game, tol=1e-9, max_iter=5000):
    """Value iteration for zero-sum stochastic games (Shapley 1953).
    Returns (V[s], policies[s]=(x,y)), V is the value for player 0."""
    assert game.N == 2
    V = np.zeros(game.S)
    for _ in range(max_iter):
        newV = np.zeros(game.S)
        for s in range(game.S):
            if not game.terminal[s]:
                newV[s] = solve_zero_sum(game.R[s, :, :, 0] + game.gamma * game.P[s] @ V)[0]
        done = np.abs(newV - V).max() < tol
        V = newV
        if done:
            break
    pol = [solve_zero_sum(game.R[s, :, :, 0] + game.gamma * game.P[s] @ V)[1:] for s in range(game.S)]
    return V, pol


# --------------------------------------------------------------------------- #
# 4. Agents
# --------------------------------------------------------------------------- #


class Agent:
    name = "agent"

    def bind(self, game, me, rng):
        self.game, self.me, self.rng = game, me, rng
        self.nA = game.nA[me]
        self.reset()

    def reset(self): ...
    def new_episode(self): ...
    def act(self, s): raise NotImplementedError
    def observe(self, s, acts, r, s2, done): ...

    def _argmax(self, q):
        return int(self.rng.choice(np.flatnonzero(q >= q.max() - 1e-12)))


class RandomAgent(Agent):
    name = "Random"

    def act(self, s): return int(self.rng.integers(self.nA))


class Always(Agent):
    def __init__(self, a=0):
        self.a, self.name = a, f"Always({a})"

    def act(self, s): return self.a


class QLearner(Agent):
    """Independent tabular Q-learning (treats the other players as environment)."""
    name = "Q-Learner"

    def __init__(self, alpha=0.2, eps=1.0, eps_min=0.02, eps_decay=0.9995, gamma=None):
        self.alpha, self.eps0, self.eps_min, self.eps_decay, self.g = alpha, eps, eps_min, eps_decay, gamma

    def reset(self):
        self.Q = np.zeros((self.game.S, self.nA))
        self.eps = self.eps0
        self.gamma = self.game.gamma if self.g is None else self.g

    def act(self, s):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.nA))
        return self._argmax(self.Q[s])

    def observe(self, s, acts, r, s2, done):
        a = acts[self.me]
        target = r[self.me] + (0.0 if done else self.gamma * self.Q[s2].max())
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def policy(self): return self.Q.argmax(1)


class MinimaxQ(Agent):
    """Minimax-Q (Littman 1994) for 2-player zero-sum stochastic games."""
    name = "Minimax-Q"

    def __init__(self, alpha=0.3, alpha_decay=0.9999, alpha_min=0.02, eps=0.3, eps_min=0.05, eps_decay=0.9998):
        self.a0, self.ad, self.am = alpha, alpha_decay, alpha_min
        self.e0, self.em, self.ed = eps, eps_min, eps_decay

    def reset(self):
        g = self.game
        assert g.N == 2, "Minimax-Q needs exactly 2 players"
        nO = g.nA[1 - self.me]
        self.Q = np.zeros((g.S, self.nA, nO))
        self.V = np.zeros(g.S)
        self.pi = np.full((g.S, self.nA), 1.0 / self.nA)
        self.alpha, self.eps = self.a0, self.e0

    def act(self, s):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.nA))
        p = np.clip(self.pi[s], 0, None)
        return int(self.rng.choice(self.nA, p=p / p.sum()))

    def observe(self, s, acts, r, s2, done):
        a, o = acts[self.me], acts[1 - self.me]
        target = r[self.me] + (0.0 if done else self.game.gamma * self.V[s2])
        self.Q[s, a, o] += self.alpha * (target - self.Q[s, a, o])
        self.V[s], self.pi[s], _ = solve_zero_sum(self.Q[s])
        self.alpha = max(self.am, self.alpha * self.ad)
        self.eps = max(self.em, self.eps * self.ed)


class RegretMatcher(Agent):
    """Per-state regret matching on the stage game (2 players, known payoffs)."""
    name = "RegretMatching"

    def reset(self):
        self.reg = np.zeros((self.game.S, self.nA))
        self.avg = np.zeros((self.game.S, self.nA))
        self._p = None

    def strategy(self, s):
        pos = np.maximum(self.reg[s], 0)
        return pos / pos.sum() if pos.sum() > 0 else np.full(self.nA, 1.0 / self.nA)

    def act(self, s):
        self._p = self.strategy(s)
        return int(self.rng.choice(self.nA, p=self._p))

    def observe(self, s, acts, r, s2, done):
        a, o = acts[self.me], acts[1 - self.me]
        u = self.game.R[s, :, o, 0] if self.me == 0 else self.game.R[s, o, :, 1]
        self.reg[s] += u - u[a]
        self.avg[s] += self._p

    def average_policy(self, s):
        t = self.avg[s].sum()
        return self.avg[s] / t if t > 0 else np.full(self.nA, 1.0 / self.nA)


class _Reactive(Agent):
    """Base for classic iterated-dilemma strategies (action 0 = cooperate, 1 = defect)."""

    def new_episode(self): self.last_me = self.last_opp = None

    def reset(self): self.new_episode()

    def observe(self, s, acts, r, s2, done):
        self.last_me, self.last_opp = acts[self.me], acts[1 - self.me]


class TitForTat(_Reactive):
    name = "TitForTat"

    def act(self, s): return 0 if self.last_opp is None else self.last_opp


class GrimTrigger(_Reactive):
    name = "Grim"

    def new_episode(self): super().new_episode(); self.betrayed = False

    def observe(self, s, acts, r, s2, done):
        super().observe(s, acts, r, s2, done)
        self.betrayed |= acts[1 - self.me] == 1

    def act(self, s): return 1 if self.betrayed else 0


class Pavlov(_Reactive):
    name = "Pavlov"

    def act(self, s): return 0 if self.last_me is None or self.last_me == self.last_opp else 1


# --------------------------------------------------------------------------- #
# 5. Simulation
# --------------------------------------------------------------------------- #


@dataclass
class Result:
    returns: np.ndarray  # (episodes, N) undiscounted return per episode
    freq: np.ndarray     # (episodes, N, maxA) action frequencies per episode


def play(game, agents, episodes=1, horizon=100, seed=0, fresh=True):
    """Run `episodes` episodes of `horizon` steps. Agents learn online.
    fresh=False keeps the agents' learned state (continue training)."""
    assert len(agents) == game.N
    rng = np.random.default_rng(seed)
    if fresh:
        for i, a in enumerate(agents):
            a.bind(game, i, np.random.default_rng(rng.integers(2**32)))
    ret = np.zeros((episodes, game.N))
    freq = np.zeros((episodes, game.N, max(game.nA)))
    for ep in range(episodes):
        s = game.reset(rng)
        for a in agents:
            a.new_episode()
        steps = 0
        for _ in range(horizon):
            acts = tuple(a.act(s) for a in agents)
            s2, r, done = game.step(acts, rng)
            for a in agents:
                a.observe(s, acts, r, s2, done)
            ret[ep] += r
            for i, ai in enumerate(acts):
                freq[ep, i, ai] += 1
            steps += 1
            s = s2
            if done:
                break
        freq[ep] /= max(steps, 1)
    return Result(ret, freq)


def tournament(game, factories, episodes=40, horizon=100, seed=0):
    """Round robin for a symmetric 2-player game. M[i,j] = mean per-step reward of i vs j."""
    names = list(factories)
    k = len(names)
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            res = play(game, [factories[names[i]](), factories[names[j]]()], episodes, horizon, seed + i * k + j)
            M[i, j] = res.returns[:, 0].mean() / horizon
    return names, M


def format_matrix(names, M, fmt="{:6.2f}"):
    w = max(len(n) for n in names)
    lines = [" " * w + " | " + " ".join(f"{n[:6]:>6}" for n in names) + " |  mean"]
    for n, row in zip(names, M):
        lines.append(f"{n:<{w}} | " + " ".join(fmt.format(v) for v in row) + f" | {row.mean():5.2f}")
    return "\n".join(lines)


def replicator(M, x0, steps=2000, dt=0.02):
    """Replicator dynamics x' = x (Mx - x.Mx) for a symmetric game M, RK4.
    Returns trajectory (steps+1, k)."""
    M = np.asarray(M, float)

    def f(x):
        u = M @ x
        return x * (u - x @ u)

    x = np.asarray(x0, float)
    x = x / x.sum()
    tr = np.empty((steps + 1, len(x)))
    tr[0] = x
    for t in range(steps):
        k1 = f(x)
        k2 = f(x + dt / 2 * k1)
        k3 = f(x + dt / 2 * k2)
        k4 = f(x + dt * k3)
        x = np.clip(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0, None)
        x /= x.sum()
        tr[t + 1] = x
    return tr


# --------------------------------------------------------------------------- #
# 6. Game library
# --------------------------------------------------------------------------- #

PD = np.array([[3., 0.], [5., 1.]])          # 0 = cooperate, 1 = defect
STAG = np.array([[4., 0.], [3., 3.]])        # 0 = stag,      1 = hare
CHICKEN = np.array([[0., -1.], [1., -10.]])  # 0 = swerve,    1 = straight
PENNIES = np.array([[1., -1.], [-1., 1.]])


def rps_matrix(win=1.0, lose=-1.0):
    return np.array([[0, lose, win], [win, 0, lose], [lose, win, 0]], float)


def matrix_game(A, B=None, gamma=0.0, name="matrix"):
    """One-state stochastic game from a bimatrix (B=None -> zero-sum)."""
    A = np.asarray(A, float)
    B = -A if B is None else np.asarray(B, float)
    m, n = A.shape
    P = np.ones((1, m, n, 1))
    R = np.stack([A, B], -1)[None]
    return StochasticGame(P, R, gamma, name=name)


def repeated(A, B=None, gamma=0.95, name="repeated"):
    """Repeated bimatrix game with memory-1 state = last joint action."""
    A = np.asarray(A, float)
    B = A.T.copy() if B is None else np.asarray(B, float)
    m, n = A.shape
    S = m * n
    P = np.zeros((S, m, n, S))
    R = np.zeros((S, m, n, 2))
    for s in range(S):
        for a in range(m):
            for b in range(n):
                P[s, a, b, a * n + b] = 1.0
                R[s, a, b] = (A[a, b], B[a, b])
    return StochasticGame(P, R, gamma, name=name)


def prisoners_dilemma(gamma=0.95): return repeated(PD, gamma=gamma, name="repeated-PD")
def stag_hunt(gamma=0.95): return repeated(STAG, gamma=gamma, name="repeated-stag-hunt")
def chicken(gamma=0.95): return repeated(CHICKEN, gamma=gamma, name="repeated-chicken")
def matching_pennies(): return matrix_game(PENNIES, name="matching-pennies")
def rock_paper_scissors(): return matrix_game(rps_matrix(), name="RPS")


def commons(levels=6, p_grow=0.5, p_revive=0.1, gamma=0.95):
    """Tragedy of the commons. State = resource level 0..levels-1.
    Action 0 = harvest 1 (conserve), 1 = harvest 2 (greedy). Harvest is capped by the
    stock and split proportionally. Remaining stock regrows by +1 or +2 (p_grow);
    a collapsed stock (0) revives to 1 only with p_revive."""
    L = levels
    P = np.zeros((L, 2, 2, L))
    R = np.zeros((L, 2, 2, 2))
    take = (1, 2)
    for s in range(L):
        for a in range(2):
            for b in range(2):
                h = np.array([take[a], take[b]], float)
                H = min(h.sum(), s)
                R[s, a, b] = H * h / h.sum()
                rem = s - H
                if rem <= 0:
                    P[s, a, b, 0] += 1 - p_revive
                    P[s, a, b, min(1, L - 1)] += p_revive
                else:
                    P[s, a, b, int(min(L - 1, rem + 1))] += 1 - p_grow
                    P[s, a, b, int(min(L - 1, rem + 2))] += p_grow
    return StochasticGame(P, R, gamma, start=np.eye(L)[L - 1], name="commons")


def random_game(S=3, A0=2, A1=2, zero_sum=True, gamma=0.9, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.full(S, 0.5), size=(S, A0, A1))
    r = rng.uniform(-1, 1, (S, A0, A1))
    R = np.stack([r, -r if zero_sum else rng.uniform(-1, 1, (S, A0, A1))], -1)
    return StochasticGame(P, R, gamma, name="random")


# --------------------------------------------------------------------------- #
# CLI: selftest + demo
# --------------------------------------------------------------------------- #


def selftest():
    ok = 0

    def check(name, cond):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + name)
        ok += bool(cond)
        assert cond, name

    v, x, y = solve_zero_sum(rps_matrix())
    check("simplex: RPS value 0, uniform", abs(v) < 1e-9 and np.allclose(x, 1 / 3) and np.allclose(y, 1 / 3))
    v, x, y = solve_zero_sum([[3, -1], [-2, 1]])
    check("simplex: 2x2 value 1/7", abs(v - 1 / 7) < 1e-9 and np.allclose(x.sum(), 1) and np.allclose(y.sum(), 1))
    rng = np.random.default_rng(1)
    good = True
    for _ in range(200):
        M = rng.normal(size=(rng.integers(2, 6), rng.integers(2, 6)))
        v, x, y = solve_zero_sum(M)
        good &= (x @ M).min() > v - 1e-7 and (M @ y).max() < v + 1e-7 and (x >= -1e-9).all()
    check("simplex: 200 random games satisfy minimax", good)
    eq = nash_equilibria(PD, PD.T)
    check("nash: PD has unique (D,D)", len(eq) == 1 and eq[0][0][1] == 1 and eq[0][1][1] == 1)
    check("nash: chicken has 3 equilibria", len(nash_equilibria(CHICKEN, CHICKEN.T)) == 3)
    g = random_game(3, 2, 2, seed=3, gamma=0.8)
    V, _ = shapley(g)
    ag = [MinimaxQ(), MinimaxQ()]
    play(g, ag, episodes=60, horizon=400, seed=1)
    err = np.abs(ag[0].V - V).max()
    check(f"minimax-Q converges to Shapley value (err={err:.3f})", err < 0.25)
    names, M = tournament(prisoners_dilemma(), {"TFT": TitForTat, "ALLD": lambda: Always(1), "ALLC": lambda: Always(0)},
                          episodes=3, horizon=50)
    check("tournament: TFT-TFT=3, ALLD-ALLC=5", abs(M[0, 0] - 3) < 1e-9 and abs(M[1, 2] - 5) < 1e-9)
    tr = replicator(rps_matrix(), [0.5, 0.3, 0.2], steps=500)
    check("replicator stays on simplex", np.allclose(tr.sum(1), 1) and (tr >= 0).all())
    print(f"{ok} checks passed")


def demo():
    t0 = time.time()
    print(f"stochgame {__version__}  (numpy {np.__version__})\n")
    print("== 1. Matrix games")
    v, x, y = solve_zero_sum(rps_matrix(1, -2))
    print(f"RPS(win 1, lose -2): value={v:+.3f}  x={np.round(x, 3)}")
    for nm, A in [("Prisoner's dilemma", PD), ("Chicken", CHICKEN), ("Stag hunt", STAG)]:
        eqs = nash_equilibria(A, A.T)
        print(f"{nm}: {len(eqs)} Nash eq ->", [(np.round(a, 2).tolist(), np.round(b, 2).tolist()) for a, b in eqs])

    print("\n== 2. Zero-sum stochastic game: Shapley vs Minimax-Q")
    g = random_game(4, 3, 3, seed=7, gamma=0.85)
    V, _ = shapley(g)
    ag = [MinimaxQ(), MinimaxQ()]
    play(g, ag, episodes=100, horizon=400, seed=0)
    print("Shapley V :", np.round(V, 3))
    print("Minimax-Q :", np.round(ag[0].V, 3))

    print("\n== 3. Tournament, repeated prisoner's dilemma (mean reward / step)")
    f = {"ALLC": lambda: Always(0), "ALLD": lambda: Always(1), "TFT": TitForTat, "Grim": GrimTrigger,
         "Pavlov": Pavlov, "Random": RandomAgent, "QLearn": QLearner}
    names, M = tournament(prisoners_dilemma(), f, episodes=60, horizon=100)
    print(format_matrix(names, M))
    tr = replicator(M, np.ones(len(names)), steps=4000, dt=0.05)
    print("replicator end state:", {n: round(float(p), 2) for n, p in zip(names, tr[-1]) if p > 0.01})

    print("\n== 4. Commons: two independent Q-learners")
    g = commons()
    ag = [QLearner(), QLearner()]
    r1 = play(g, ag, episodes=300, horizon=50, seed=0)
    print(f"conserve rate  first 30 ep: {r1.freq[:30, :, 0].mean():.2f}   last 30 ep: {r1.freq[-30:, :, 0].mean():.2f}")
    print(f"mean return    first 30 ep: {r1.returns[:30].mean():.1f}   last 30 ep: {r1.returns[-30:].mean():.1f}")
    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    {"demo": demo, "selftest": selftest}.get(cmd, lambda: print(__doc__))()

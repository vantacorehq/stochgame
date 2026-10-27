import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stochgame as sg  # noqa: E402


def test_selftest():
    sg.selftest()


def test_nash_gap_zero_at_pd_equilibrium():
    x = y = np.array([0.0, 1.0])  # (defect, defect)
    assert sg.nash_gap(sg.PD, sg.PD.T, x, y) == pytest.approx(0.0)


def test_shapley_matches_matrix_game_value():
    V, _ = sg.shapley(sg.matrix_game(sg.PENNIES))
    assert V[0] == pytest.approx(0.0, abs=1e-9)


def test_invalid_transition_rows_rejected():
    P = np.full((1, 2, 2, 1), 2.0)  # rows sum to 2
    R = np.zeros((1, 2, 2, 2))
    with pytest.raises(ValueError):
        sg.StochasticGame(P, R)


def test_play_shapes_and_zero_sum():
    g = sg.matrix_game(sg.PENNIES)
    res = sg.play(g, [sg.RandomAgent(), sg.RandomAgent()], episodes=5, horizon=10)
    assert res.returns.shape == (5, 2)
    assert res.freq.shape == (5, 2, 2)
    assert np.allclose(res.returns.sum(1), 0.0)


def test_play_is_reproducible():
    def run():
        return sg.play(sg.commons(), [sg.QLearner(), sg.QLearner()],
                       episodes=20, horizon=30, seed=42).returns
    assert np.array_equal(run(), run())


def test_custom_agent_in_tournament():
    class Sulky(sg.Agent):
        name = "Sulky"

        def reset(self): self.bad = 0
        def new_episode(self): self.bad = 0
        def act(self, s): return 1 if self.bad >= 2 else 0
        def observe(self, s, acts, r, s2, done):
            self.bad = self.bad + 1 if acts[1 - self.me] == 1 else 0

    names, M = sg.tournament(sg.prisoners_dilemma(), {"TFT": sg.TitForTat, "Sulky": Sulky},
                             episodes=3, horizon=20)
    assert names == ["TFT", "Sulky"]
    assert M.shape == (2, 2)
    assert M[0, 0] == pytest.approx(3.0)  # TFT vs TFT: full cooperation

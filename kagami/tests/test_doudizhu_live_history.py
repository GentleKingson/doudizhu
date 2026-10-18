import nanopokerlib as npl
import numpy as np
import pytest


ROOT_HISTORY = (
    "L+33567888TTJQKKKA2|P1+4556677789TTKA22B|"
    "P2+3344456999JJJQQA2|L+QAR"
)


def test_committed_combo_history_round_trips_through_reconstruction() -> None:
    game = npl.doudizhu
    envs, step = game.reconstruct_envs([ROOT_HISTORY], seed = 0)
    history = ROOT_HISTORY
    for action, suffix in [("888?", ""), ("888.", ""), ("8885", "|L-5888"),
                           ("PASS", "|P1-PASS"), ("JJJ?", ""), ("JJJ.", ""),
                           ("JJJ3", "|P2-3JJJ")]:
        index = envs.get_available_action_strs()[0].index(action)
        assert step.available_actions.weights[0, index] > 0
        tokens = np.asarray(step.available_actions.tokens[0:1, index], dtype = np.int64)
        step = envs.step(game.actions_from_np(tokens))
        assert envs.advance_history_strs([history]) == [history + suffix]
        history += suffix

    envs, step = game.reconstruct_envs([history], seed = 0)
    index = envs.get_available_action_strs()[0].index("PASS")
    tokens = np.asarray(step.available_actions.tokens[0:1, index], dtype = np.int64)
    envs.step(game.actions_from_np(tokens))
    with pytest.raises(RuntimeError, match = "expected prior canonical history"):
        envs.advance_history_strs([history.replace("|L-5888", "|L-8885")])
    assert envs.advance_history_strs([history]) == [history + "|L-PASS"]

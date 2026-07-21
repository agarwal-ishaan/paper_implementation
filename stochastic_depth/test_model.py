import pytest

from model import survival_probabilities


def test_survival_probabilities_length():
    probs = survival_probabilities(18)
    assert len(probs) == 18


def test_survival_probabilities_last_equals_p_L():
    probs = survival_probabilities(18, p_L=0.5)
    assert probs[-1] == pytest.approx(0.5)


def test_survival_probabilities_first_close_to_one():
    probs = survival_probabilities(18, p_L=0.5)
    assert probs[0] == pytest.approx(1 - (1 / 18) * 0.5)


def test_survival_probabilities_monotonically_decreasing():
    probs = survival_probabilities(18)
    assert all(probs[i] > probs[i + 1] for i in range(len(probs) - 1))


def test_survival_probabilities_all_one_when_p_L_is_one():
    probs = survival_probabilities(10, p_L=1.0)
    assert all(p == pytest.approx(1.0) for p in probs)

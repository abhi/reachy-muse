import pytest

from bench.voice_verify import calibrate, eer, rates


def test_error_rates_count_boundary_acceptances():
    assert rates([.8, .6, .4, .2], [.7, .5, .3, .1], .5) == {"far": .5, "frr": .5}
    assert eer([.8, .6, .4, .2], [.7, .5, .3, .1]) == .5
    assert eer([.9, .8], [.2, .1]) == 0.


def test_calibration_uses_only_calibration_impostors_and_margin():
    assert calibrate([.8, .7], [.21, .35]) == .37
    assert rates([.8, .7], [.21, .35], .37) == {"far": 0., "frr": 0.}


def test_empty_evaluation_is_an_error():
    with pytest.raises(ValueError, match="needs genuine and impostor"):
        rates([], [.2], .5)

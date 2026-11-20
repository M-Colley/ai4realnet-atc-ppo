"""Decision rule of scripts.final_experiments (pre-registered in docs/solution/EXPERIMENTS_2026-11.md)."""

from scripts import final_experiments as fe
from scripts.seed_repeat_pooled import METRICS


def _effects(ma_fail=(0.0, -1.0, 1.0), score=(0.0, -0.1, 0.1), viol=(0.0, -0.01, 0.01)):
    e = {k: dict(mean=0.0, ci=[-1.0, 1.0]) for k in METRICS}
    e[fe.MA_FAIL] = dict(mean=ma_fail[0], ci=list(ma_fail[1:]))
    e[fe.SCORE] = dict(mean=score[0], ci=list(score[1:]))
    for k in fe.VIOLATIONS:
        e[k] = dict(mean=viol[0], ci=list(viol[1:]))
    return e


NO_SA_CHANGE = dict(new=3, fixed=3, p=1.0)


def test_null_result_does_not_qualify():
    r = fe.decide(_effects(), NO_SA_CHANGE, _effects())
    assert r == dict(rule1=False, rule2=True, rule3=None, qualifies=False)


def test_gain_via_ma_failures_needs_confirmation():
    gain = _effects(ma_fail=(-10.0, -18.0, -2.0))
    assert fe.decide(gain, NO_SA_CHANGE, _effects(ma_fail=(-4.0, -12.0, 3.0)))["qualifies"]
    # confirmation stream with the opposite sign
    assert not fe.decide(gain, NO_SA_CHANGE, _effects(ma_fail=(2.0, -6.0, 10.0)))["qualifies"]
    # confirmation stream whose score CI lies entirely below 0
    bad = _effects(ma_fail=(-4.0, -12.0, 3.0), score=(-0.3, -0.5, -0.1))
    assert not fe.decide(gain, NO_SA_CHANGE, bad)["qualifies"]
    # not yet evaluated on the confirmation stream
    assert fe.decide(gain, NO_SA_CHANGE, None)["rule3"] is None


def test_gain_via_score_requires_no_more_ma_failures():
    assert fe.decide(_effects(score=(0.2, 0.05, 0.35)), NO_SA_CHANGE, _effects(score=(0.1, -0.1, 0.3)))["qualifies"]
    worse_ma = _effects(ma_fail=(2.0, -6.0, 10.0), score=(0.2, 0.05, 0.35))
    assert not fe.decide(worse_ma, NO_SA_CHANGE, _effects())["rule1"]


def test_safety_regression_blocks():
    gain = dict(ma_fail=(-10.0, -18.0, -2.0))
    more_violations = _effects(**gain, viol=(0.01, 0.002, 0.02))
    assert not fe.decide(more_violations, NO_SA_CHANGE, _effects())["rule2"]
    more_sa_failures = dict(new=12, fixed=2, p=0.04)
    assert not fe.decide(_effects(**gain), more_sa_failures, _effects())["rule2"]
    fewer_sa_failures = dict(new=2, fixed=12, p=0.04)
    assert fe.decide(_effects(**gain), fewer_sa_failures, _effects(ma_fail=(-1.0, -5.0, 3.0)))["rule2"]

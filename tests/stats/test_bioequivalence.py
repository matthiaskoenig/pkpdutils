import numpy as np
import pytest
from scipy.stats import t as student_t

from pkpdutils import Route, Timecourses, nca
from pkpdutils.stats import ParameterSample
from pkpdutils.stats.bioequivalence import (
    BEParameter,
    BEResult,
    Design,
    bioequivalence,
    tost,
)
from pkpdutils.stats.ratio import ratio

N = 12
RNG = np.random.default_rng(21)
SUBJECTS = np.array([f"s{i:02d}" for i in range(N)])
# sequence RT: reference in period 1, test in period 2; TR the other way round
SEQUENCE = np.array(["RT"] * 6 + ["TR"] * 6)
PERIOD_TEST = np.where(SEQUENCE == "RT", 2, 1)
PERIOD_REF = 3 - PERIOD_TEST
SUBJECT_EFFECT = RNG.normal(0, 0.3, N)
PERIOD_EFFECT = np.array([0.0, 0.2])  # period 2 raises the log exposure by 0.2
TRUE_LOG_RATIO = np.log(0.95)


def log_value(treatment_effect: float, period: np.ndarray) -> np.ndarray:
    return (
        np.log(100.0)
        + SUBJECT_EFFECT
        + treatment_effect
        + PERIOD_EFFECT[period - 1]
        + RNG.normal(0, 0.08, N)
    )


TEST_VALUES = np.exp(log_value(TRUE_LOG_RATIO, PERIOD_TEST))
REF_VALUES = np.exp(log_value(0.0, PERIOD_REF))


def crossover_samples() -> tuple[ParameterSample, ParameterSample]:
    test = ParameterSample(
        values=TEST_VALUES,
        labels=SUBJECTS,
        coords={"period": PERIOD_TEST, "sequence": SEQUENCE},
        name="auc_inf_obs",
        unit="mg*hr/l",
    )
    perm = RNG.permutation(N)
    reference = ParameterSample(
        values=REF_VALUES[perm],
        labels=SUBJECTS[perm],
        coords={"period": PERIOD_REF[perm], "sequence": SEQUENCE[perm]},
        name="auc_inf_obs",
        unit="mg*hr/l",
    )
    return test, reference


def ols_crossover() -> tuple[float, float, float]:
    """Treatment effect, its standard error and the residual variance of the ANOVA with subject, period and treatment effects."""
    y = np.concatenate([np.log(TEST_VALUES), np.log(REF_VALUES)])
    treatment = np.concatenate([np.ones(N), np.zeros(N)])
    period2 = np.concatenate([PERIOD_TEST == 2, PERIOD_REF == 2]).astype(float)
    subject = np.concatenate([np.eye(N), np.eye(N)])[:, 1:]  # drop one subject dummy
    x = np.column_stack([np.ones(2 * N), treatment, period2, subject])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    mse = residuals @ residuals / (2 * N - x.shape[1])
    cov = mse * np.linalg.inv(x.T @ x)
    return float(beta[1]), float(np.sqrt(cov[1, 1])), float(mse)


def test_crossover_equals_the_anova() -> None:
    test, reference = crossover_samples()
    res = tost(test, reference)
    assert isinstance(res, BEParameter) and res.design is Design.CROSSOVER
    effect, se, mse = ols_crossover()
    assert res.log_ratio == pytest.approx(effect)
    assert res.se_log == pytest.approx(se)
    assert res.df == N - 2
    assert res.cv_intra == pytest.approx(np.sqrt(np.expm1(mse)))
    tq = student_t.ppf(0.95, N - 2)
    assert (res.ci_low, res.ci_high) == pytest.approx(
        (np.exp(effect - tq * se), np.exp(effect + tq * se))
    )
    assert res.gmr == pytest.approx(np.exp(effect))
    # the period effect of 0.2 is detected (p = 7e-4 for this seed), the sequence (carryover) effect is absent (p = 0.68)
    assert res.p_period < 0.01 and res.p_sequence > 0.05
    assert res.n_test == N and res.n_reference == N and res.name == "auc_inf_obs"


def test_tost_p_values_and_verdict() -> None:
    test, reference = crossover_samples()
    res = tost(test, reference)
    t_low = (res.log_ratio - np.log(0.8)) / res.se_log
    t_high = (np.log(1.25) - res.log_ratio) / res.se_log
    assert res.p_lower == pytest.approx(student_t.sf(t_low, N - 2))
    assert res.p_upper == pytest.approx(student_t.sf(t_high, N - 2))
    assert res.p_value == max(res.p_lower, res.p_upper)
    assert res.bioequivalent == (res.ci_low >= 0.8 and res.ci_high <= 1.25)
    assert res.bioequivalent == (res.p_value < 0.05)
    wide = tost(test, reference, limits=(0.5, 2.0))
    assert wide.bioequivalent and wide.limits == (0.5, 2.0)
    narrow = tost(test, reference, limits=(0.99, 1.01))
    assert not narrow.bioequivalent


def test_paired_and_parallel_designs() -> None:
    test = ParameterSample(values=TEST_VALUES, labels=SUBJECTS)
    reference = ParameterSample(values=REF_VALUES, labels=SUBJECTS)
    paired = tost(test, reference)
    assert paired.design is Design.PAIRED
    r = ratio(test, reference)
    assert (paired.gmr, paired.ci_low, paired.ci_high) == (r.gmr, r.ci_low, r.ci_high)
    d = np.log(TEST_VALUES) - np.log(REF_VALUES)
    assert paired.cv_intra == pytest.approx(np.sqrt(np.expm1(d.var(ddof=1) / 2)))
    assert np.isnan(paired.p_period)
    parallel = tost(
        ParameterSample(values=TEST_VALUES), ParameterSample(values=REF_VALUES[:10])
    )
    assert parallel.design is Design.PARALLEL and np.isnan(parallel.cv_intra)
    r2 = ratio(
        ParameterSample(values=TEST_VALUES), ParameterSample(values=REF_VALUES[:10])
    )
    assert parallel.se_log == pytest.approx(r2.se_log)
    forced = tost(test, reference, design=Design.PARALLEL)
    assert forced.design is Design.PARALLEL
    summary = tost(
        ParameterSample(mean=95.0, sd=20.0, n=12),
        ParameterSample(mean=100.0, sd=22.0, n=12),
    )
    assert summary.design is Design.PARALLEL and np.isfinite(summary.p_value)


def test_crossover_validation() -> None:
    test, reference = crossover_samples()
    bad_period = ParameterSample(
        values=REF_VALUES,
        labels=SUBJECTS,
        coords={"period": PERIOD_TEST, "sequence": SEQUENCE},
    )
    with pytest.raises(ValueError, match="period"):
        tost(test, bad_period)
    one_sequence = ParameterSample(
        values=REF_VALUES,
        labels=SUBJECTS,
        coords={"period": PERIOD_REF, "sequence": np.array(["RT"] * N)},
    )
    with pytest.raises(ValueError, match="sequence"):
        tost(test, one_sequence)
    same_period_test = ParameterSample(
        values=TEST_VALUES,
        labels=SUBJECTS,
        coords={"period": np.full(N, 2), "sequence": SEQUENCE},
    )
    same_period_reference = ParameterSample(
        values=REF_VALUES,
        labels=SUBJECTS,
        coords={"period": np.full(N, 1), "sequence": SEQUENCE},
    )
    with pytest.raises(ValueError, match="period 2 in one sequence"):
        tost(same_period_test, same_period_reference)
    with pytest.raises(ValueError, match="crossover"):
        tost(
            ParameterSample(values=TEST_VALUES),
            ParameterSample(values=REF_VALUES),
            design=Design.CROSSOVER,
        )
    with pytest.raises(ValueError, match="limits"):
        tost(test, reference, limits=(1.25, 0.8))


def batch(values: np.ndarray, period: np.ndarray) -> Timecourses:
    time = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])
    curves = np.stack([v / 5 * np.exp(-0.2 * time) for v in values])
    return Timecourses.from_arrays(
        time,
        curves,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={
            "individual": SUBJECTS,
            "period": ("individual", period),
            "sequence": ("individual", SEQUENCE),
        },
        dose={"amount": np.full(N, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
    )


def test_bioequivalence_of_nca_results() -> None:
    test = nca(batch(TEST_VALUES, PERIOD_TEST))
    reference = nca(batch(REF_VALUES, PERIOD_REF))
    res = bioequivalence(test, reference)
    assert isinstance(res, BEResult)
    assert set(res.parameters) == {"auc_inf_obs", "cmax"}
    assert res["auc_inf_obs"].design is Design.CROSSOVER
    assert res.bioequivalent == all(p.bioequivalent for p in res.parameters.values())
    direct = tost(
        test.sample("cmax", "individual"), reference.sample("cmax", "individual")
    )
    assert res["cmax"].gmr == direct.gmr
    df = res.to_dataframe()
    assert list(df["parameter"]) == ["auc_inf_obs", "cmax"]
    assert {
        "gmr",
        "ci_low",
        "ci_high",
        "bioequivalent",
        "p_value",
        "design",
        "cv_intra",
    } <= set(df.columns)
    with pytest.raises(ValueError, match="not a variable"):
        bioequivalence(test, reference, parameters=["auc_inf"])

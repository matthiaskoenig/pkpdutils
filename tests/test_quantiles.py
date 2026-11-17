"""The quantiles of the rust kernel against scipy, over a dense grid.

The engine gate of the host compares at ``rel=1e-6``, so a quantile which is good to 1e-8 only
would pass it. These sweeps are the guard of the accuracy: 1000 probabilities log-spaced over
[1e-300, 0.5] and their reflections ``1 - p``, at every degree of freedom of a grid.

scipy is not right everywhere. Its ``t.isf`` answers ``-inf`` or ``0.0`` for some of the
probabilities below 1e-6 and its ``t.sf`` underflows to 0 once ``t**2`` overflows, so the
tail below 1e-6 is held to two independent checks instead of to scipy's own quantile: the
round trip through ``scipy.stats.t.sf`` where scipy can evaluate it, and the closed form of
the power law of the tail where it cannot.
"""

import numpy as np
import pkpdutils_core
import pytest
from numpy.typing import NDArray
from scipy import special, stats

_MISSING = [
    name
    for name in ("normal_isf", "student_t_isf", "student_t_sf")
    if not hasattr(pkpdutils_core, name)
]
if _MISSING:
    # the floor `pkpdutils-core>=0.3.0` admits the published 0.3.0 wheel, which has none of the
    # quantiles and reports the same version as a build which has them, so the attributes are
    # what is asked and not the version. Without this the module raises while it is collected
    # and pytest runs nothing at all.
    pytest.skip(
        f"pkpdutils_core {getattr(pkpdutils_core, '__version__', '?')} lacks {', '.join(_MISSING)}: rebuild the "
        "kernel into this environment (`maturin develop`) or raise the pkpdutils-core floor",
        allow_module_level=True,
    )

# the extension module carries no stubs, so ty cannot see its functions
normal_isf = pkpdutils_core.normal_isf  # ty: ignore[missing-attribute]
student_t_isf = pkpdutils_core.student_t_isf  # ty: ignore[missing-attribute]
student_t_sf = pkpdutils_core.student_t_sf  # ty: ignore[missing-attribute]

DEGREES_OF_FREEDOM = (1.0, 1.5, 2.0, 2.5, 3.0, 7.3, 10.0, 30.0, 100.0, 1000.0, 10000.0)
# the probabilities scipy and the kernel can both answer at 1e-12
ACCURATE_FROM = 1e-6


def _grid() -> NDArray[np.float64]:
    lower = np.logspace(-300, np.log10(0.5), 1000)
    return np.concatenate([lower, 1.0 - lower[::-1]])


def _tail_power_law(p: NDArray[np.float64], df: float) -> NDArray[np.float64]:
    """The t with ``P(T > t) = p`` for ``t**2 >> df``, where ``sf = A df**((df-1)/2) t**-df``.

    ``A = Gamma((df+1)/2) / (sqrt(pi df) Gamma(df/2))`` times ``df**((df+1)/2) / df``; the
    relative error of the dropped terms is ``df / t**2``, below 1e-200 for ``t > 1e100``.
    """
    log_a = (
        special.gammaln((df + 1.0) / 2.0)
        - special.gammaln(df / 2.0)
        - 0.5 * np.log(np.pi * df)
    )
    log_t = (log_a + 0.5 * (df - 1.0) * np.log(df) - np.log(p)) / df
    return np.exp(log_t)


def test_the_kernel_normal_quantile_matches_scipy_over_a_dense_grid() -> None:
    grid = _grid()
    got = normal_isf(grid)
    expected = stats.norm.isf(grid)
    # one assertion for everything, the infinities (p rounded to 1.0) and the zero at 1/2
    # included: assert_allclose requires an infinity where scipy has one
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)
    assert np.isneginf(got[grid == 1.0]).all()
    assert (got[grid == 0.5] == 0.0).all()
    assert np.isfinite(got[grid < 1.0]).all()
    # the sweep actually reaches the far tail and its reflection
    assert got[0] > 37.0
    assert got[grid < 1.0][-1] < -7.0


@pytest.mark.parametrize("df", DEGREES_OF_FREEDOM)
def test_the_kernel_student_t_quantile_matches_scipy_over_a_dense_grid(
    df: float,
) -> None:
    grid = _grid()
    got = student_t_isf(grid, df)

    # the stated domain: scipy answers and agrees at 1e-12, the infinities and the zero included
    inside = grid >= ACCURATE_FROM
    np.testing.assert_allclose(
        got[inside], stats.t.isf(grid[inside], df), rtol=1e-12, atol=0.0
    )
    assert np.isneginf(got[grid == 1.0]).all()
    assert (got[grid == 0.5] == 0.0).all()

    # below 1e-6 scipy's quantile is unreliable, so the answer has to invert the survival
    # function instead: scipy's where it does not underflow, the power law of the tail otherwise
    tail = grid < ACCURATE_FROM
    p, t = grid[tail], got[tail]
    assert np.isfinite(t).all()
    assert (t > 0.0).all()
    moderate = t < 1e100
    np.testing.assert_allclose(
        stats.t.sf(t[moderate], df), p[moderate], rtol=1e-10, atol=0.0
    )
    far = ~moderate
    if df <= 2.0:
        assert far.any(), (
            "the heaviest tails reach t > 1e100, where scipy's sf has underflowed"
        )
    np.testing.assert_allclose(
        t[far], _tail_power_law(p[far], df), rtol=1e-11, atol=0.0
    )


@pytest.mark.parametrize(
    ("df", "p", "expected"),
    [
        # mpmath at 60 digits for the exact floats: python scripts/reference_t_isf.py 1e-300 3 10 2.5
        (3.0, 1e-300, 1.0331108360446529e100),
        (10.0, 1e-300, 2.5645257189481978e30),
        (2.5, 1e-300, 8.765437882279992e119),
    ],
)
def test_the_kernel_student_t_quantile_at_the_smallest_probability_matches_mpmath(
    df: float, p: float, expected: float
) -> None:
    assert student_t_isf(p, df) == pytest.approx(expected, rel=1e-12)


def test_the_kernel_student_t_survival_function_matches_scipy_over_a_dense_grid() -> (
    None
):
    t = np.concatenate(
        [-np.logspace(-8, 100, 400)[::-1], [0.0], np.logspace(-8, 100, 400)]
    )
    for df in DEGREES_OF_FREEDOM:
        got = student_t_sf(t, df)
        expected = stats.t.sf(t, df)
        if df == 1.0:
            # the Cauchy closed form; scipy's t.sf is off by up to 3e-9 relative for |t| < 1e-5
            # here (0.4999999952568131 at t = 1e-8 against 0.49999999681690114, the mpmath value)
            positive = t > 0.0
            expected = np.where(
                positive,
                np.arctan(1.0 / np.where(positive, t, 1.0)) / np.pi,
                0.5 - np.arctan(t) / np.pi,
            )
        # scipy's sf underflows to 0 where t**2 overflows and loses the tail below 1e-300 to
        # the denormals; compare where it holds a normal number
        usable = expected > 1e-290
        assert usable.sum() > 300
        np.testing.assert_allclose(got[usable], expected[usable], rtol=1e-12, atol=0.0)
        # and where scipy gives nothing the survival function is still a probability which never
        # increases with t, strictly while it is neither rounded to 1 nor to the denormals
        assert (got >= 0.0).all()
        assert (got <= 1.0).all()
        step = np.diff(got)
        assert (step <= 0.0).all()
        assert (step[(got[1:] > 1e-300) & (got[:-1] < 0.999)] < 0.0).all()


@pytest.mark.parametrize("df", DEGREES_OF_FREEDOM)
def test_the_kernel_student_t_quantile_is_monotone_below_the_accurate_domain(
    df: float,
) -> None:
    p = np.logspace(-300, -6, 20000)
    t = student_t_isf(p, df)
    assert np.isfinite(t).all()
    # a larger p has a strictly smaller quantile
    assert (np.diff(t) < 0.0).all()
    # and it is not merely ordered: the survival function of the kernel gives p back
    np.testing.assert_allclose(student_t_sf(t, df), p, rtol=1e-9, atol=0.0)


def test_the_kernel_student_t_survival_function_is_exact_for_the_cauchy_near_zero() -> (
    None
):
    # mpmath at 40 digits: 1/2 - atan(1e-8) / pi
    assert student_t_sf(1e-8, 1.0) == pytest.approx(0.49999999681690113816, rel=1e-15)
    assert student_t_sf(-1e-8, 1.0) == pytest.approx(0.50000000318309886184, rel=1e-15)


def test_the_kernel_quantiles_accept_an_array_and_return_its_shape() -> None:
    p = np.array([1e-10, 0.025, 0.5, 0.975])
    normal = normal_isf(p)
    quantile = student_t_isf(p, 7.3)
    survival = student_t_sf(quantile, 7.3)
    for result in (normal, quantile, survival):
        assert isinstance(result, np.ndarray)
        assert result.shape == p.shape
        assert result.dtype == np.float64
    # element by element the array call is the scalar call
    for i, value in enumerate(p):
        assert normal[i] == normal_isf(float(value))
        assert quantile[i] == student_t_isf(float(value), 7.3)
        assert survival[i] == student_t_sf(float(quantile[i]), 7.3)
    np.testing.assert_allclose(normal, stats.norm.isf(p), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(quantile, stats.t.isf(p, 7.3), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(survival, p, rtol=1e-12, atol=0.0)

    # a float stays a float, an integer is read as one
    assert type(normal_isf(0.025)) is float
    assert student_t_isf(0.025, 10.0) == pytest.approx(2.2281388519862753, rel=1e-12)
    assert normal_isf(1) == float("-inf")

    # empty and strided input
    assert normal_isf(np.empty(0)).shape == (0,)
    strided = np.linspace(0.1, 0.9, 9)[::2]
    np.testing.assert_array_equal(
        student_t_isf(strided, 5.0),
        student_t_isf(np.ascontiguousarray(strided), 5.0),
    )
    readonly = p.copy()
    readonly.setflags(write=False)
    np.testing.assert_array_equal(normal_isf(readonly), normal)

    # outside the domain: NaN, never an exception, and below one degree of freedom as well
    assert np.isnan(normal_isf(1.5))
    assert np.isnan(student_t_isf(0.025, 0.5))
    assert np.isnan(student_t_isf(np.array([-0.1, 2.0]), 3.0)).all()

    # anything which is not a float or a 1-d float array is a TypeError
    with pytest.raises(TypeError, match="1-d numpy array"):
        normal_isf(np.ones((2, 2)))
    with pytest.raises(TypeError, match="1-d numpy array"):
        student_t_isf("0.025", 3.0)

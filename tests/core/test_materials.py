"""Material model for the solver: reluctivity against flux density."""

import math
import re
import warnings
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
import scipy.optimize
import yaml

from polecraft.core.materials import (
    AIR,
    COPPER,
    MU0,
    BHCurve,
    Conductor,
    CoreLossCoefficients,
    CoreLossData,
    FrohlichSteel,
    LinearMaterial,
    LossRangeWarning,
    Magnet,
    load_core_loss,
    load_material,
    material_names,
)

# A typical non-oriented electrical steel, in tesla and A/m. The values are
# illustrative, not a datasheet.
STEEL_B = np.array(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
     1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
)
STEEL_H = np.array(
    [0.0, 30.0, 40.0, 47.0, 53.0, 59.0, 66.0, 74.0, 84.0, 97.0,
     115.0, 141.0, 186.0, 288.0, 660.0, 1820.0, 3900.0, 6900.0, 11000.0]
)

# Tables that are valid but awkward for a smooth interpolation.
COARSE = ([0.0, 1.0, 1.5], [0.0, 100.0, 1000.0])  # sharp knee, three points
FLATTENING = ([0.0, 0.5, 1.0, 1.5], [0.0, 50.0, 2000.0, 2010.0])  # slope falls at the end
UNEVEN = ([0.0, 0.01, 1.2, 1.21, 2.0], [0.0, 2.0, 150.0, 400.0, 50000.0])
TABLES = {
    "steel": (STEEL_B, STEEL_H),
    "coarse": COARSE,
    "flattening": FLATTENING,
    "uneven": UNEVEN,
}


@pytest.fixture
def steel() -> BHCurve:
    return BHCurve(STEEL_B, STEEL_H)


@pytest.fixture(params=sorted(TABLES))
def curve(request: pytest.FixtureRequest) -> BHCurve:
    return BHCurve(*TABLES[request.param])


def dense(curve: BHCurve, beyond: float = 1.5, count: int = 4001) -> np.ndarray:
    """Flux densities from zero to past the end of the curve's table."""
    return np.linspace(0.0, beyond * curve.b[-1], count)


# Linear materials


def test_mu0_is_the_classical_value() -> None:
    assert MU0 == pytest.approx(4e-7 * math.pi, rel=1e-15)


def test_air_reluctivity_is_one_over_mu0() -> None:
    assert AIR.reluctivity(np.array([0.0, 0.5, 2.0])) == pytest.approx(1.0 / MU0, rel=1e-15)


def test_copper_is_as_permeable_as_air() -> None:
    assert COPPER.reluctivity(np.array([1.0])) == pytest.approx(1.0 / MU0, rel=1e-15)


def test_magnet_uses_its_recoil_permeability() -> None:
    magnet = LinearMaterial(1.05)
    assert magnet.reluctivity(np.array([0.0, 1.2])) == pytest.approx(1.0 / (1.05 * MU0))


def test_linear_reluctivity_does_not_depend_on_flux_density() -> None:
    material = LinearMaterial(500.0)
    b = np.linspace(0.0, 3.0, 7)
    assert np.all(material.reluctivity(b) == material.reluctivity(b)[0])
    assert np.all(material.reluctivity_derivative(b) == 0.0)


def test_linear_material_says_it_is_linear() -> None:
    assert AIR.linear
    assert LinearMaterial(1.05).linear


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_linear_material_rejects_impossible_permeability(value: float) -> None:
    with pytest.raises(ValueError, match="relative permeability"):
        LinearMaterial(value)


@pytest.mark.parametrize("shape", [(), (5,), (3, 4)])
def test_linear_results_have_the_shape_of_the_input(shape: tuple[int, ...]) -> None:
    b = np.full(shape, 0.7)
    assert AIR.reluctivity(b).shape == shape
    assert AIR.reluctivity_derivative(b).shape == shape


# BH curve: table points


def test_curve_says_it_is_not_linear(steel: BHCurve) -> None:
    assert not steel.linear


def test_field_strength_reproduces_the_table(curve: BHCurve) -> None:
    assert curve.field_strength(curve.b) == pytest.approx(curve.h, rel=1e-12, abs=1e-12)


def test_reluctivity_reproduces_the_table(curve: BHCurve) -> None:
    b, h = curve.b[1:], curve.h[1:]
    assert curve.reluctivity(b) == pytest.approx(h / b, rel=1e-12)


def test_reluctivity_is_h_over_b_everywhere(curve: BHCurve) -> None:
    b = dense(curve)[1:]
    assert curve.reluctivity(b) * b == pytest.approx(curve.field_strength(b), rel=1e-12)


def test_table_is_copied(steel: BHCurve) -> None:
    b, h = STEEL_B.copy(), STEEL_H.copy()
    own = BHCurve(b, h)
    b[3], h[3] = 99.0, 99.0
    assert own.field_strength(STEEL_B) == pytest.approx(steel.field_strength(STEEL_B))


# BH curve: shape


def test_field_strength_rises_with_flux_density(curve: BHCurve) -> None:
    h = curve.field_strength(dense(curve))
    assert np.all(np.diff(h) > 0.0)


def test_flux_density_rises_with_field_strength_between_table_points(steel: BHCurve) -> None:
    # The same statement read as B(H), on a grid much finer than the table.
    b = np.linspace(0.0, STEEL_B[-1], 20001)
    h = steel.field_strength(b)
    assert np.all(np.diff(h) > 0.0)
    assert h[0] == 0.0
    assert h[-1] == pytest.approx(STEEL_H[-1])


def test_reluctivity_is_positive_and_finite(curve: BHCurve) -> None:
    nu = curve.reluctivity(dense(curve))
    assert np.all(np.isfinite(nu))
    assert np.all(nu > 0.0)


def test_reluctivity_never_exceeds_that_of_air_for_steel(steel: BHCurve) -> None:
    assert np.all(steel.reluctivity(dense(steel)) < 1.0 / MU0)


def test_reluctivity_at_zero_is_the_limit_from_above(curve: BHCurve) -> None:
    at_zero = curve.reluctivity(np.array(0.0))
    just_above = curve.reluctivity(np.array(1e-9 * curve.b[1]))
    assert at_zero == pytest.approx(just_above, rel=1e-6)


def test_reluctivity_is_continuous_at_the_table_points(curve: BHCurve) -> None:
    b = curve.b[1:]
    step = 1e-9 * curve.b[-1]
    below, above = curve.reluctivity(b - step), curve.reluctivity(b + step)
    assert below == pytest.approx(above, rel=1e-5)


def test_saturating_steel_gets_stiffer(steel: BHCurve) -> None:
    b = np.linspace(1.2, 2.5, 200)
    assert np.all(np.diff(steel.reluctivity(b)) > 0.0)
    assert np.all(steel.reluctivity_derivative(b) > 0.0)


# BH curve: derivative


def test_derivative_matches_a_finite_difference(curve: BHCurve) -> None:
    # Midpoints of a fine grid, so no sample sits on a table point, where the
    # second derivative jumps and a central difference loses accuracy.
    grid = dense(curve, count=3001)
    b = 0.5 * (grid[:-1] + grid[1:])
    step = 1e-6 * curve.b[-1]
    numeric = (curve.reluctivity(b + step) - curve.reluctivity(b - step)) / (2 * step)
    analytic = curve.reluctivity_derivative(b)
    scale = np.max(np.abs(analytic))
    assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-7 * scale)


def test_derivative_is_continuous_at_the_table_points(curve: BHCurve) -> None:
    b = curve.b[1:]
    step = 1e-9 * curve.b[-1]
    below = curve.reluctivity_derivative(b - step)
    above = curve.reluctivity_derivative(b + step)
    scale = np.max(np.abs(curve.reluctivity_derivative(dense(curve))))
    assert below == pytest.approx(above, rel=1e-5, abs=1e-6 * scale)


def test_derivative_at_zero_is_the_limit_from_above(curve: BHCurve) -> None:
    at_zero = curve.reluctivity_derivative(np.array(0.0))
    just_above = curve.reluctivity_derivative(np.array(1e-9 * curve.b[1]))
    scale = np.max(np.abs(curve.reluctivity_derivative(dense(curve))))
    assert np.isfinite(at_zero)
    assert at_zero == pytest.approx(just_above, rel=1e-5, abs=1e-6 * scale)


def test_differential_reluctivity_is_the_slope_of_the_curve(curve: BHCurve) -> None:
    # dH/dB = nu + B dnu/dB, which is what the Newton iteration uses.
    grid = dense(curve, count=3001)
    b = 0.5 * (grid[:-1] + grid[1:])
    step = 1e-6 * curve.b[-1]
    slope = (curve.field_strength(b + step) - curve.field_strength(b - step)) / (2 * step)
    built = curve.reluctivity(b) + b * curve.reluctivity_derivative(b)
    assert built == pytest.approx(slope, rel=1e-5)
    assert np.all(built > 0.0)


# BH curve: known answers


def test_straight_line_table_is_linear_within_the_table() -> None:
    b = np.array([0.0, 0.4, 1.0, 1.7])
    curve = BHCurve(b, b / (1000.0 * MU0))
    sample = np.linspace(0.0, 1.7, 50)
    assert curve.reluctivity(sample) == pytest.approx(1.0 / (1000.0 * MU0), rel=1e-12)
    assert curve.reluctivity_derivative(sample) == pytest.approx(0.0, abs=1e-9 / MU0)


def slope_of(curve: BHCurve, b: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """dH/dB by central difference."""
    return (curve.field_strength(b + step) - curve.field_strength(b - step)) / (2.0 * step)


def test_tail_joins_the_table_with_the_same_value_and_slope(curve: BHCurve) -> None:
    end = curve.b[-1]
    below, above = end - 1e-12, end + 1e-12
    assert float(curve.field_strength(above)) == pytest.approx(float(curve.field_strength(below)))
    # d(nu)/dB = (dH/dB - nu) / B, so a jump in slope would show here.
    left = float(curve.reluctivity_derivative(below))
    right = float(curve.reluctivity_derivative(above))
    assert right == pytest.approx(left, rel=1e-6, abs=1e-9 / MU0)


def test_beyond_the_table_the_slope_approaches_that_of_air(curve: BHCurve) -> None:
    end = curve.b[-1]
    b = end + np.array([0.01, 0.1, 0.5, 2.0, 30.0])
    slopes = slope_of(curve, b) * MU0  # 1 for air
    assert np.all(np.diff(slopes) >= -1e-6)  # never falls on the way to air
    assert slopes[0] < slopes[2] and np.all(slopes <= 1.0 + 1e-6)
    assert slopes[-1] == pytest.approx(1.0, abs=1e-6)


def test_beyond_the_table_polarisation_levels_off_at_saturation(curve: BHCurve) -> None:
    b = curve.b[-1] + np.array([0.0, 0.05, 0.2, 1.0, 20.0])
    j = b - MU0 * curve.field_strength(b)
    assert np.all(np.diff(j) > 0.0)
    assert j[-1] == pytest.approx(curve.saturation, abs=1e-9)


def test_curve_is_monotonic_far_beyond_the_table(curve: BHCurve) -> None:
    b = np.linspace(0.0, curve.b[-1] + 5.0, 20001)
    assert np.all(np.diff(curve.field_strength(b)) > 0.0)


def test_default_saturation_follows_the_stated_rule(steel: BHCurve) -> None:
    end_slope = float(slope_of(steel, np.array([STEEL_B[-1] - 1e-4]), 1e-6)[0])
    rule = STEEL_B[-1] - MU0 * STEEL_H[-1] + 0.1 * (1.0 - MU0 * end_slope)
    assert steel.saturation == pytest.approx(rule, rel=1e-3)


def test_given_saturation_is_used(steel: BHCurve) -> None:
    curve = BHCurve(STEEL_B, STEEL_H, saturation=2.05)
    assert curve.saturation == 2.05
    b = np.array([30.0])
    assert float(b[0] - MU0 * curve.field_strength(b)[0]) == pytest.approx(2.05, abs=1e-9)
    # Within the table nothing changes.
    inside = np.linspace(0.0, STEEL_B[-1], 101)
    assert curve.field_strength(inside) == pytest.approx(steel.field_strength(inside), rel=1e-14)


@pytest.mark.parametrize("saturation", [1.0, 1.786, float("nan"), float("inf")])
def test_saturation_below_the_last_row_is_rejected(saturation: float) -> None:
    with pytest.raises(ValueError, match="saturation polarisation must be above"):
        BHCurve(STEEL_B, STEEL_H, saturation=saturation)


def test_table_already_as_soft_as_air_continues_straight() -> None:
    b = np.array([0.0, 1.0, 2.0])
    curve = BHCurve(b, b / MU0)
    assert curve.saturation is None
    assert curve.reluctivity(np.array([3.0, 5.0])) == pytest.approx(1.0 / MU0, rel=1e-12)


@pytest.mark.parametrize(
    ("j", "h"),
    [
        # The last interval is at the slope of air; the one before is not.
        ([0.0, 1.0, 1.8, 1.9, 1.9], [0.0, 100.0, 1e4, 1e5, 2e5]),
        # The last interval is near the slope of air (0.83 of it).
        ([0.0, 1.0, 1.8, 1.85], [0.0, 100.0, 1e4, 2e5]),
    ],
    ids=["at the slope of air", "near the slope of air"],
)
def test_curve_is_never_stiffer_than_air_beyond_the_table(j: list, h: list) -> None:
    # The interpolation's own end slope is above 1 / mu0 for these tables
    # (1.17 / mu0 for the first), and a tail that started there would keep
    # asking for more field than air does, however far it went.
    polarisation, field = np.array(j), np.array(h)
    curve = BHCurve(polarisation + MU0 * field, field)
    assert curve.field_strength(curve.b) == pytest.approx(field, rel=1e-12, abs=1e-9)
    end = curve.b[-1]
    beyond = end + np.concatenate([[0.0, 1e-9, 1e-6], np.linspace(1e-3, 30.0, 3001)])
    step = 1e-7
    slope = (curve.field_strength(beyond + step) - curve.field_strength(beyond)) / step
    assert np.all(MU0 * slope <= 1.0 + 1e-6)
    assert MU0 * slope[-1] == pytest.approx(1.0, abs=1e-6)
    # d(nu)/dB = (dH/dB - nu) / B gives the slope without differencing.
    exact = curve.reluctivity(beyond) + beyond * curve.reluctivity_derivative(beyond)
    assert np.all(MU0 * exact <= 1.0 + 1e-12)
    # So the polarisation never falls beyond the last row.
    after = beyond - MU0 * curve.field_strength(beyond)
    assert np.all(np.diff(after) >= -1e-12)
    assert after[-1] >= polarisation[-1] - 1e-12
    assert np.all(np.diff(curve.field_strength(np.linspace(0.0, end + 5.0, 20001))) > 0.0)


# Stacking factor


@pytest.mark.parametrize("factor", [0.9, 0.95, 0.97])
def test_stacking_factor_scales_the_polarisation_at_the_table_points(
    steel: BHCurve, factor: float
) -> None:
    stack = steel.stacked(factor)
    assert np.array_equal(stack.h, steel.h)
    assert stack.b == pytest.approx(factor * steel.b + (1.0 - factor) * MU0 * steel.h, rel=1e-15)
    assert stack.b - MU0 * stack.h == pytest.approx(factor * (steel.b - MU0 * steel.h), abs=1e-12)


def test_stacking_factor_scales_the_effective_flux_density_everywhere(steel: BHCurve) -> None:
    factor = 0.95
    stack = steel.stacked(factor)
    b = np.linspace(0.05, 2.6, 300)  # through the table and into the tail
    h = steel.field_strength(b)
    effective = factor * b + (1.0 - factor) * MU0 * h
    assert stack.field_strength(effective) == pytest.approx(h, rel=0.02)
    assert stack.saturation == pytest.approx(factor * steel.saturation, rel=1e-12)


def test_stacking_factor_of_one_changes_nothing(steel: BHCurve) -> None:
    b = np.linspace(0.0, 3.0, 61)
    assert steel.stacked(1.0).field_strength(b) == pytest.approx(steel.field_strength(b), rel=1e-12)


def test_stacked_steel_needs_more_field_for_the_same_flux_density(steel: BHCurve) -> None:
    b = np.linspace(0.2, 2.0, 10)
    assert np.all(steel.stacked(0.95).field_strength(b) > steel.field_strength(b))


@pytest.mark.parametrize("factor", [0.0, -0.5, 1.01, float("nan")])
def test_impossible_stacking_factor_is_rejected(steel: BHCurve, factor: float) -> None:
    with pytest.raises(ValueError, match="stacking factor"):
        steel.stacked(factor)


def test_two_point_table() -> None:
    curve = BHCurve([0.0, 1.0], [0.0, 200.0])
    b = np.array([0.0, 0.3, 0.7, 1.0])
    assert curve.field_strength(b) == pytest.approx(200.0 * b)
    assert curve.reluctivity(b) == pytest.approx(200.0)


# BH curve: arrays in and out


@pytest.mark.parametrize("shape", [(), (5,), (3, 4)])
def test_curve_results_have_the_shape_of_the_input(steel: BHCurve, shape: tuple[int, ...]) -> None:
    b = np.full(shape, 1.35)
    for result in (
        steel.field_strength(b),
        steel.reluctivity(b),
        steel.reluctivity_derivative(b),
    ):
        assert result.shape == shape
        assert result.dtype == np.float64


def test_curve_accepts_plain_numbers_and_lists(steel: BHCurve) -> None:
    assert float(steel.reluctivity(1.0)) == pytest.approx(115.0)
    assert steel.reluctivity([1.0, 1.5]) == pytest.approx([115.0, 1820.0 / 1.5])


def test_curve_does_not_change_its_input(steel: BHCurve) -> None:
    b = np.linspace(0.0, 2.5, 11)
    before = b.copy()
    steel.reluctivity(b)
    steel.reluctivity_derivative(b)
    assert np.array_equal(b, before)


def test_mixed_ranges_in_one_call_match_separate_calls(steel: BHCurve) -> None:
    b = np.array([2.4, 0.0, 0.05, 1.8, 0.1, 1.45, 3.0])
    together = steel.reluctivity(b)
    alone = np.array([float(steel.reluctivity(value)) for value in b])
    assert together == pytest.approx(alone, rel=1e-15)


# BH curve: bad tables


@pytest.mark.parametrize(
    ("b", "h", "message"),
    [
        ([0.0, 1.0], [0.0, 100.0, 200.0], "same number"),
        ([[0.0, 1.0]], [[0.0, 100.0]], "one-dimensional"),
        ([0.0], [0.0], "at least two"),
        ([0.1, 1.0], [10.0, 100.0], "start at B = 0, H = 0"),
        ([0.0, 1.0], [5.0, 100.0], "start at B = 0, H = 0"),
        ([0.0, 1.0, 1.0, 1.5], [0.0, 100.0, 200.0, 300.0], "flux density must increase"),
        ([0.0, 1.0, 0.9, 1.5], [0.0, 100.0, 200.0, 300.0], "flux density must increase"),
        ([0.0, 1.0, 1.2, 1.5], [0.0, 100.0, 100.0, 300.0], "field strength must increase"),
        ([0.0, 1.0, 1.2, 1.5], [0.0, 100.0, 90.0, 300.0], "field strength must increase"),
        ([0.0, 1.0, math.nan], [0.0, 100.0, 200.0], "finite"),
        ([0.0, 1.0, 1.5], [0.0, 100.0, math.inf], "finite"),
        ([0.0, 1.0, 2.0], [0.0, 100.0, 3e6], "polarisation J = B - mu0 H must not fall"),
        ([0.0, 1.0, 1.5, 1.6], [0.0, 100.0, 1000.0, 2e5], "polarisation J = B - mu0 H must not"),
    ],
)
def test_bad_tables_are_rejected(b: list, h: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BHCurve(b, h)


def test_bad_table_message_names_the_row() -> None:
    with pytest.raises(ValueError, match="row 3"):
        BHCurve([0.0, 1.0, 0.9, 1.5], [0.0, 100.0, 200.0, 300.0])


def test_falling_polarisation_message_names_the_row() -> None:
    # Row 3 asks for 3e6 A/m to gain 1 T; air alone needs 1 / mu0 = 0.8e6.
    with pytest.raises(ValueError, match=r"row 3 \(-1.76991 T\) is below row 2 \(0.999874 T\)"):
        BHCurve([0.0, 1.0, 2.0], [0.0, 100.0, 3e6])


def test_polarisation_that_stays_level_is_accepted() -> None:
    # Fully saturated rows follow the slope of air; B - mu0 H then differs
    # from row to row only by rounding.
    j = np.array([0.0, 1.5, 1.9, 1.9, 1.9])
    h = np.array([0.0, 1000.0, 1e5, 2e5, 3e5])
    curve = BHCurve(j + MU0 * h, h)
    assert curve.saturation is None


# Magnets


def test_magnet_keeps_its_remanence_and_recoil_permeability() -> None:
    magnet = Magnet(remanence=1.2, relative_permeability=1.05)
    assert magnet.remanence == 1.2
    assert magnet.linear
    assert magnet.reluctivity(np.array([0.0, 0.9])) == pytest.approx(1.0 / (1.05 * MU0))
    assert np.all(magnet.reluctivity_derivative(np.array([0.0, 0.9])) == 0.0)


def test_magnet_with_no_remanence_is_allowed() -> None:
    assert Magnet(remanence=0.0).remanence == 0.0


def test_magnet_recoil_permeability_has_a_usual_default() -> None:
    assert Magnet(remanence=1.0).relative_permeability == pytest.approx(1.05)


@pytest.mark.parametrize("remanence", [-0.1, math.nan, math.inf])
def test_magnet_rejects_impossible_remanence(remanence: float) -> None:
    with pytest.raises(ValueError, match="remanence"):
        Magnet(remanence=remanence)


def test_magnet_rejects_impossible_permeability() -> None:
    with pytest.raises(ValueError, match="relative permeability"):
        Magnet(remanence=1.2, relative_permeability=0.0)


# Analytical saturation law


@pytest.fixture
def law() -> FrohlichSteel:
    return FrohlichSteel(saturation=2.0, half_field=300.0)


def test_law_is_b_equals_mu0_h_plus_a_saturating_polarisation(law: FrohlichSteel) -> None:
    h = np.array([0.0, 10.0, 300.0, 5e3, 1e5, 1e7])
    expected = MU0 * h + 2.0 * h / (300.0 + h)
    assert law.flux_density(h) == pytest.approx(expected, rel=1e-14)
    # Half the saturation polarisation is reached at the half field.
    assert law.flux_density(300.0) - MU0 * 300.0 == pytest.approx(1.0)


def test_law_field_strength_inverts_flux_density(law: FrohlichSteel) -> None:
    h = np.concatenate([[0.0], np.logspace(-3, 8, 300)])
    assert law.field_strength(law.flux_density(h)) == pytest.approx(h, rel=1e-9, abs=1e-12)


def test_law_says_it_is_not_linear(law: FrohlichSteel) -> None:
    assert not law.linear


def test_law_reluctivity_is_h_over_b(law: FrohlichSteel) -> None:
    b = np.linspace(0.01, 4.0, 400)
    assert law.reluctivity(b) * b == pytest.approx(law.field_strength(b), rel=1e-12)


def test_law_starts_at_its_initial_permeability(law: FrohlichSteel) -> None:
    # For small fields B = (mu0 + Js / a) H.
    initial = 1.0 / (MU0 + 2.0 / 300.0)
    assert law.reluctivity(0.0) == pytest.approx(initial, rel=1e-12)
    assert law.reluctivity(1e-9) == pytest.approx(initial, rel=1e-6)


def test_law_ends_with_the_slope_of_air(law: FrohlichSteel) -> None:
    b = np.array([50.0, 50.001])
    slope = np.diff(law.field_strength(b)) / np.diff(b)
    assert slope[0] == pytest.approx(1.0 / MU0, rel=1e-3)
    assert np.all(law.reluctivity(np.linspace(0.0, 100.0, 500)) < 1.0 / MU0)


def test_law_gets_stiffer_all_the_way(law: FrohlichSteel) -> None:
    b = np.linspace(0.0, 5.0, 2000)
    assert np.all(np.diff(law.reluctivity(b)) > 0.0)
    assert np.all(law.reluctivity_derivative(b) > 0.0)
    assert np.all(np.diff(law.field_strength(b)) > 0.0)


def test_law_derivative_matches_a_finite_difference(law: FrohlichSteel) -> None:
    b = np.linspace(0.02, 4.0, 700)
    step = 1e-6
    numeric = (law.reluctivity(b + step) - law.reluctivity(b - step)) / (2.0 * step)
    analytic = law.reluctivity_derivative(b)
    assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-7 * np.abs(analytic).max())


def test_law_derivative_at_zero_is_the_limit_from_above(law: FrohlichSteel) -> None:
    at_zero = law.reluctivity_derivative(0.0)
    assert np.isfinite(at_zero)
    assert at_zero == pytest.approx(law.reluctivity_derivative(1e-6), rel=1e-4)


@pytest.mark.parametrize("shape", [(), (5,), (3, 4)])
def test_law_results_have_the_shape_of_the_input(law: FrohlichSteel, shape: tuple) -> None:
    b = np.full(shape, 1.35)
    for result in (law.field_strength(b), law.reluctivity(b), law.reluctivity_derivative(b)):
        assert result.shape == shape


@pytest.mark.parametrize(
    ("saturation", "half_field"), [(0.0, 300.0), (-1.0, 300.0), (2.0, 0.0), (math.nan, 300.0)]
)
def test_law_rejects_impossible_values(saturation: float, half_field: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        FrohlichSteel(saturation=saturation, half_field=half_field)


# Material files


def test_placeholder_steel_loads_as_a_bh_curve() -> None:
    steel = load_material("generic_35A_placeholder")
    assert isinstance(steel, BHCurve)
    assert not steel.linear
    assert steel.b[-1] > 2.2  # reaches well past any flux density in a motor


def test_placeholder_steel_is_the_analytical_law_and_nothing_else() -> None:
    # The file is not steel data: every row comes from one formula.
    steel = load_material("generic_35A_placeholder")
    law = FrohlichSteel(saturation=2.0, half_field=300.0)
    assert steel.b == pytest.approx(law.flux_density(steel.h), abs=5e-7)


def test_placeholder_steel_follows_the_law_between_its_rows() -> None:
    steel = load_material("generic_35A_placeholder")
    law = FrohlichSteel(saturation=2.0, half_field=300.0)
    b = np.linspace(0.05, 2.3, 400)
    # Between rows the table is interpolated, so it is close, not equal:
    # within 4 % below the first row and within 1 % above it.
    assert steel.field_strength(b) == pytest.approx(law.field_strength(b), rel=0.05)
    above = b > steel.b[1]
    assert steel.field_strength(b[above]) == pytest.approx(law.field_strength(b[above]), rel=0.01)


def test_placeholder_steel_ends_close_to_the_slope_of_air() -> None:
    steel = load_material("generic_35A_placeholder")
    b = np.array([3.0, 3.5])
    slope = np.diff(steel.field_strength(b)) / np.diff(b)
    assert slope[0] == pytest.approx(1.0 / MU0, rel=0.02)


def test_placeholder_file_says_it_is_not_real_data() -> None:
    text = resources.files("polecraft.data").joinpath(
        "materials", "generic_35A_placeholder.yaml"
    ).read_text(encoding="utf-8")
    assert "not real steel data" in text.lower()
    assert "placeholder" in text.lower()


def test_steels_in_the_library() -> None:
    steels = [n for n in material_names() if isinstance(load_material(n), BHCurve)]
    assert steels == ["M270-35A", "generic_35A_placeholder"]


@pytest.mark.parametrize("name", material_names())
def test_every_library_entry_loads_with_its_fields(name: str) -> None:
    data = yaml.safe_load(
        resources.files("polecraft.data")
        .joinpath("materials", f"{name}.yaml")
        .read_text(encoding="utf-8")
    )
    assert data["name"] == name
    assert data["source"].strip() and data["description"].strip()
    expected = {"steel": BHCurve, "magnet": Magnet, "conductor": Conductor}[data["kind"]]
    assert isinstance(load_material(name), expected)


def test_library_has_each_kind() -> None:
    kinds = {type(load_material(n)) for n in material_names()}
    assert kinds == {BHCurve, Magnet, Conductor}


def test_magnet_entry_gives_its_remanence_and_permeability() -> None:
    magnet = load_material("generic_NdFeB_placeholder")
    assert magnet.remanence == 1.2
    assert magnet.relative_permeability == 1.05


def test_magnet_placeholder_says_it_is_not_a_datasheet_value() -> None:
    text = resources.files("polecraft.data").joinpath(
        "materials", "generic_NdFeB_placeholder.yaml"
    ).read_text(encoding="utf-8")
    assert "not a datasheet value" in text.lower()


@pytest.fixture
def grade() -> Magnet:
    return load_material("generic_NdFeB_placeholder")


def test_magnet_at_20_c_has_its_file_values(grade: Magnet) -> None:
    cold = grade.at_temperature(20.0)
    assert cold.remanence == 1.2
    assert cold.relative_permeability == 1.05
    assert cold.coercivity == pytest.approx(1.2 / (1.05 * MU0), rel=1e-12)
    assert grade.coercivity == cold.coercivity


def test_magnet_coefficient_comes_from_the_file(grade: Magnet) -> None:
    assert grade.remanence_coefficient == -0.0012
    assert grade.temperature == 20.0


def test_placeholder_magnet_gives_no_intrinsic_coercivity(grade: Magnet) -> None:
    # There is no datasheet to take one from.
    assert grade.intrinsic_coercivity is None
    assert grade.intrinsic_coercivity_coefficient is None


@pytest.mark.parametrize("temperature", [-40.0, 0.0, 80.0, 150.0])
def test_remanence_follows_its_coefficient(grade: Magnet, temperature: float) -> None:
    moved = grade.at_temperature(temperature)
    assert moved.remanence == pytest.approx(
        1.2 * (1.0 - 0.0012 * (temperature - 20.0)), rel=1e-12
    )
    assert moved.temperature == temperature


def test_remanence_at_80_c_by_hand(grade: Magnet) -> None:
    # 60 K above 20 C at -0.12 %/K is 7.2 % down: 1.2 x 0.928 = 1.1136 T.
    assert grade.at_temperature(80.0).remanence == pytest.approx(1.1136, rel=1e-12)


@pytest.mark.parametrize("temperature", [-40.0, -20.0, 0.0, 20.0, 60.0, 80.0, 100.0, 150.0])
def test_recoil_permeability_does_not_change_with_temperature(
    grade: Magnet, temperature: float
) -> None:
    moved = grade.at_temperature(temperature)
    assert moved.relative_permeability == 1.05
    assert moved.reluctivity(0.5) == grade.reluctivity(0.5)


@pytest.mark.parametrize("temperature", [-40.0, 80.0, 150.0])
def test_coercivity_on_the_recoil_line_follows_the_remanence(
    grade: Magnet, temperature: float
) -> None:
    moved = grade.at_temperature(temperature)
    assert moved.coercivity / grade.coercivity == pytest.approx(
        moved.remanence / grade.remanence, rel=1e-12
    )
    # B = mu_r mu0 H + Br is zero at H = -Hcb.
    assert 1.05 * MU0 * -moved.coercivity + moved.remanence == pytest.approx(0.0, abs=1e-12)


def test_remanence_falls_as_the_magnet_warms_and_rises_as_it_cools(grade: Magnet) -> None:
    temperatures = [-40.0, -20.0, 20.0, 60.0, 100.0, 150.0]
    magnets = [grade.at_temperature(t) for t in temperatures]
    assert all(a.remanence > b.remanence for a, b in zip(magnets, magnets[1:]))
    assert all(a.coercivity > b.coercivity for a, b in zip(magnets, magnets[1:]))


def test_intrinsic_coercivity_is_kept_as_data_and_changes_nothing() -> None:
    plain = Magnet(remanence=1.2, remanence_coefficient=-0.001)
    full = Magnet(
        remanence=1.2,
        remanence_coefficient=-0.001,
        intrinsic_coercivity=1.5e6,
        intrinsic_coercivity_coefficient=-0.005,
    )
    hot = full.at_temperature(120.0)
    assert hot.intrinsic_coercivity == 1.5e6  # still the 20 C value
    assert hot.intrinsic_coercivity_coefficient == -0.005
    assert hot.remanence == plain.at_temperature(120.0).remanence
    assert hot.relative_permeability == plain.relative_permeability
    assert hot.coercivity == plain.at_temperature(120.0).coercivity


def test_magnet_takes_no_coefficient_for_hcb() -> None:
    with pytest.raises(TypeError, match="coercivity_coefficient"):
        Magnet(remanence=1.2, coercivity_coefficient=-0.004)


@pytest.mark.parametrize("value", [0.0, -9e5, float("nan"), float("inf")])
def test_magnet_rejects_an_impossible_intrinsic_coercivity(value: float) -> None:
    with pytest.raises(ValueError, match="intrinsic_coercivity must be a positive number"):
        Magnet(remanence=1.2, intrinsic_coercivity=value)


def test_intrinsic_coercivity_coefficient_needs_the_coercivity() -> None:
    with pytest.raises(ValueError, match="needs intrinsic_coercivity"):
        Magnet(remanence=1.2, intrinsic_coercivity_coefficient=-0.005)


def test_magnet_with_no_temperature_data_cannot_be_moved() -> None:
    with pytest.raises(ValueError, match="no temperature coefficient"):
        Magnet(remanence=1.2).at_temperature(80.0)


def test_a_warmed_magnet_is_not_warmed_again(grade: Magnet) -> None:
    with pytest.raises(ValueError, match="start from the 20 C magnet"):
        grade.at_temperature(80.0).at_temperature(100.0)


@pytest.mark.parametrize("temperature", [900.0, 1000.0, float("nan"), float("inf")])
def test_temperature_outside_the_linear_law_is_rejected(grade: Magnet, temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature must be a finite|does not hold"):
        grade.at_temperature(temperature)


@pytest.mark.parametrize("temperature", [-273.16, -300.0, -1000.0, float("-inf")])
def test_temperature_below_absolute_zero_is_rejected(grade: Magnet, temperature: float) -> None:
    with pytest.raises(ValueError, match="absolute zero|temperature must be a finite"):
        grade.at_temperature(temperature)


def test_magnet_can_be_taken_down_to_absolute_zero(grade: Magnet) -> None:
    cold = grade.at_temperature(-273.15)
    assert cold.remanence == pytest.approx(1.2 * (1.0 + 0.0012 * 293.15), rel=1e-12)
    assert cold.relative_permeability == 1.05


@pytest.mark.parametrize(
    "name", ["remanence_coefficient", "intrinsic_coercivity_coefficient", "temperature"]
)
def test_magnet_rejects_a_non_finite_temperature_value(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        Magnet(remanence=1.2, intrinsic_coercivity=1.5e6, **{name: float("nan")})


def test_magnet_reluctivity_does_not_change_with_the_coefficients(grade: Magnet) -> None:
    assert grade.reluctivity(0.5) == Magnet(1.2, 1.05).reluctivity(0.5)


def test_copper_has_the_annealed_standard_values() -> None:
    # International Annealed Copper Standard at 20 C: 58 m/(ohm mm2),
    # 0.00393 per kelvin and 8.89 g/cm3.
    copper = load_material("copper")
    assert copper.resistivity == 1.7241e-8
    assert copper.resistivity == pytest.approx(1.0 / 58e6, rel=1e-4)
    assert copper.temperature_coefficient == 0.00393
    assert copper.resistivity_at(20.0) == copper.resistivity
    assert copper.density == 8890.0


def test_copper_file_says_where_its_density_comes_from(package_dir: Path) -> None:
    text = (package_dir / "data" / "materials" / "copper.yaml").read_text(encoding="utf-8")
    assert "8.89 g/cm3" in text
    assert "NBS Handbook 100" in text


def test_copper_resistivity_rises_with_temperature() -> None:
    copper = load_material("copper")
    # 100 C is 80 K above 20 C: 1 + 0.00393 * 80 = 1.3144
    assert copper.resistivity_at(100.0) == pytest.approx(copper.resistivity * 1.3144)
    assert copper.resistivity_at(0.0) < copper.resistivity


def test_copper_resistivity_holds_down_to_where_the_line_reaches_zero() -> None:
    copper = load_material("copper")
    # 1 + 0.00393 (T - 20) is zero at T = 20 - 1 / 0.00393 = -234.45 C.
    assert copper.resistivity_at(-234.0) > 0.0
    assert copper.resistivity_at(-100.0) == pytest.approx(copper.resistivity * 0.5284)


@pytest.mark.parametrize("temperature", [-234.5, -250.0, -273.15, -1000.0])
def test_resistivity_of_zero_or_less_is_refused(temperature: float) -> None:
    copper = load_material("copper")
    with pytest.raises(ValueError, match="resistivity of zero or less; it does not hold"):
        copper.resistivity_at(temperature)


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf")])
def test_resistivity_needs_a_finite_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature must be a finite number"):
        load_material("copper").resistivity_at(temperature)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resistivity": 0.0},
        {"resistivity": float("nan")},
        {"temperature_coefficient": -0.1},
        {"density": float("inf")},
    ],
)
def test_conductor_rejects_impossible_values(kwargs: dict) -> None:
    values = {"resistivity": 1.7e-8, "temperature_coefficient": 0.004, "density": 8890.0}
    with pytest.raises(ValueError, match="positive number"):
        Conductor(**{**values, **kwargs})


def test_unknown_material_is_rejected_with_the_known_names() -> None:
    with pytest.raises(ValueError, match="M330-35A.*M270-35A.*generic_35A_placeholder"):
        load_material("M330-35A")


@pytest.mark.parametrize("name", ["../secrets", "a/b", "", "x.yaml"])
def test_material_names_cannot_be_paths(name: str) -> None:
    with pytest.raises(ValueError, match="material"):
        load_material(name)


HEAD = "name: broken\ndescription: x\nsource: y\n"
MAGNET = HEAD + "kind: magnet\nremanence_T: 1.2\nrelative_permeability: 1.05\n"
HCJ = "intrinsic_coercivity_A_per_m: 1.5e6\n"
HCJ_COEFFICIENT = "intrinsic_coercivity_temp_coefficient_per_K: -0.005\n"
CONDUCTOR = (
    HEAD
    + "kind: conductor\nresistivity_ohm_m: 1.7e-8\n"
    + "temperature_coefficient_per_K: 0.004\ndensity_kg_m3: 8890\n"
)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- just\n- a list\n", "must be a mapping"),
        (HEAD + "kind: ferrite\n", "'kind' must be one of steel, magnet, conductor"),
        ("name: other\ndescription: x\nsource: y\nkind: steel\n", "'name' must be 'broken'"),
        ("name: broken\nsource: y\nkind: steel\n", "'description' must be a non-empty text"),
        ("name: broken\ndescription: x\nkind: steel\n", "'source' must be a non-empty text"),
        ("name: broken\ndescription: x\nsource: ' '\nkind: steel\n", "'source' must be"),
        (HEAD + "kind: steel\n", "give one of 'bh_curve' and 'jh_curve'"),
        (
            HEAD + "kind: steel\nbh_curve: [[0, 0], [1, 9]]\njh_curve: [[0, 0], [1, 9]]\n",
            "give one of 'bh_curve' and 'jh_curve'",
        ),
        (HEAD + "kind: steel\njh_curve: [[0.1, 30], [0.2]]\n", "'jh_curve' must be a list of"),
        (
            HEAD + "kind: steel\njh_curve: [[0.1, 30], [0.2, 20]]\n",
            "field strength must increase",
        ),
        (
            # Too small a fall to see once mu0 H is added: B still rises.
            HEAD + "kind: steel\njh_curve: [[0.1, 30], [1.0, 100], [0.9999999999, 200]]\n",
            "'jh_curve' polarisation must not fall from row to row; row 3 ",
        ),
        (
            HEAD + "kind: steel\njh_curve: [[0.1, 30], [1.0, 100], [0.9, 200]]\n",
            "'jh_curve' polarisation must not fall",
        ),
        (
            HEAD + "kind: steel\nbh_curve: [[0, 0], [1, 100], [2, 3.0e+6]]\n",
            "polarisation J = B - mu0 H must not fall",
        ),
        (
            HEAD + "kind: steel\njh_curve: [[0.1, 30], [0.2, 40]]\nsaturation_polarisation_T: 0.1\n",
            "saturation polarisation must be above",
        ),
        (HEAD + "kind: steel\nbh_curve: [[0, 0, 0], [1, 100, 5]]\n", "'bh_curve' must be a list"),
        (HEAD + "kind: steel\nbh_curve: [[0, 0], [1, x]]\n", "'bh_curve' must be a list"),
        (
            HEAD + "kind: steel\nbh_curve: [[0, 0], [1, 100], [0.9, 200]]\n",
            "flux density must increase",
        ),
        (MAGNET.replace("remanence_T: 1.2\n", ""), "'remanence_T' \\(tesla\\) must be"),
        (MAGNET.replace("1.2", "-1.2"), "'remanence_T' \\(tesla\\) must be zero or more"),
        (MAGNET.replace("1.05", "zero"), "'relative_permeability'"),
        (MAGNET.replace("1.2", "true"), "'remanence_T'"),
        (
            MAGNET + "remanence_temp_coefficient_per_K: x\n",
            "'remanence_temp_coefficient_per_K' \\(fraction per kelvin, about 20 C\\) must be a number",
        ),
        (MAGNET + "remanence_temp_coefficient_per_K: .inf\n", "'remanence_temp_coefficient_per_K'"),
        (
            MAGNET + "coercivity_temp_coefficient_per_K: -0.004\n",
            # A datasheet's coercivity coefficient is for Hcj, not Hcb.
            "'coercivity_temp_coefficient_per_K' does not say which coercivity.*Hcj.*"
            "write it as 'intrinsic_coercivity_temp_coefficient_per_K', with "
            "'intrinsic_coercivity_A_per_m'.*Hcb, takes no coefficient",
        ),
        (
            MAGNET + HCJ_COEFFICIENT,
            "'intrinsic_coercivity_temp_coefficient_per_K' needs 'intrinsic_coercivity_A_per_m'",
        ),
        (
            MAGNET + HCJ.replace("1.5e6", "-1.5e6"),
            "'intrinsic_coercivity_A_per_m' \\(ampere per metre at 20 C\\) must be a positive",
        ),
        (MAGNET + HCJ + HCJ_COEFFICIENT.replace("-0.005", "steep"), "must be a number"),
        (MAGNET + "remanence_T: 1.3\n", "not valid YAML at line 7: field 'remanence_T' appears twice"),
        (CONDUCTOR + "density_kg_m3: 8890\n", "field 'density_kg_m3' appears twice"),
        (MAGNET.replace("1.05", "010"), "'relative_permeability'"),
        (MAGNET + "notes: [unclosed\n", "not valid YAML"),
        (CONDUCTOR.replace("1.7e-8", "0"), "'resistivity_ohm_m' \\(ohm metre at 20 C\\)"),
        (CONDUCTOR.replace("8890", ".nan"), "'density_kg_m3'"),
        (
            CONDUCTOR.replace("temperature_coefficient_per_K: 0.004\n", ""),
            "'temperature_coefficient_per_K'",
        ),
    ],
)
def test_a_bad_material_file_is_rejected(
    text: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "materials").mkdir()
    (tmp_path / "materials" / "broken.yaml").write_text(text, encoding="utf-8")
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)
    assert material_names() == ["broken"]
    with pytest.raises(ValueError, match=message):
        load_material("broken")


def library_entry(text: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Load ``text`` as the only entry of a temporary library."""
    (tmp_path / "materials").mkdir()
    (tmp_path / "materials" / "broken.yaml").write_text(text, encoding="utf-8")
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)
    return load_material("broken")


@pytest.mark.parametrize(
    ("written", "value"),
    [("2e-8", 2e-8), ("2E-8", 2e-8), ("1.7e-8", 1.7e-8), ("17e-9", 17e-9), ("+2e-8", 2e-8)],
)
def test_material_file_reads_numbers_without_a_decimal_point(
    written: str, value: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conductor = library_entry(CONDUCTOR.replace("1.7e-8", written), tmp_path, monkeypatch)
    assert conductor.resistivity == value


def test_material_file_reads_numbers_the_way_a_design_file_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    magnet = library_entry(
        MAGNET.replace("1.05", "105e-2") + "intrinsic_coercivity_A_per_m: 15e5\n",
        tmp_path,
        monkeypatch,
    )
    assert magnet.relative_permeability == 1.05
    assert magnet.intrinsic_coercivity == 1.5e6


def test_magnet_entry_with_datasheet_coercivity_fields_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = MAGNET + "remanence_temp_coefficient_per_K: -0.001\n" + HCJ + HCJ_COEFFICIENT
    magnet = library_entry(text, tmp_path, monkeypatch)
    assert magnet.remanence_coefficient == -0.001
    assert magnet.intrinsic_coercivity == 1.5e6
    assert magnet.intrinsic_coercivity_coefficient == -0.005
    assert magnet.at_temperature(120.0).relative_permeability == 1.05


def test_magnet_entry_may_give_the_remanence_coefficient_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    magnet = library_entry(
        MAGNET + "remanence_temp_coefficient_per_K: -0.001\n", tmp_path, monkeypatch
    )
    assert magnet.at_temperature(120.0).remanence == pytest.approx(1.2 * 0.9)
    assert magnet.intrinsic_coercivity is None



# M270-35A, from the Cogent Power brochure
#
# The rows below are the brochure's (Cogent 2002-11), copied from the text of
# the PDF, so that the library file is checked against something other than
# itself.

# Peak field strength in A/m at J = 0.1 to 1.8 T in steps of 0.1 T, 50 Hz.
# Page 22 up to 1.0 T, page 23 from 1.1 T.
M270_H = [
    30.0, 39.6, 46.0, 52.0, 58.2, 65.2, 73.3, 83.1, 95.5, 112,
    136, 178, 272, 596, 1700, 3880, 7160, 11600,
]
# Typical magnetic properties, page 6: J in tesla at H in A/m, 50 Hz.
M270_TYPICAL_J = {2500.0: 1.54, 5000.0: 1.65, 10000.0: 1.77}
# Specific total loss in W/kg from J = 0.1 T in steps of 0.1 T, by frequency
# in Hz.
M270_LOSS = {
    # Page 18 up to 1.0 T, page 19 from 1.1 T.
    50: [0.03, 0.07, 0.13, 0.22, 0.31, 0.43, 0.54, 0.68, 0.83, 1.01,
         1.20, 1.42, 1.70, 2.12, 2.47, 2.80, 3.05, 3.25],
    # Page 24.
    100: [0.04, 0.16, 0.34, 0.55, 0.80, 1.08, 1.38, 1.73, 2.10, 2.51,
          2.98, 3.51, 4.15, 4.97, 5.92],
    200: [0.09, 0.37, 0.79, 1.31, 1.91, 2.61, 3.39, 4.26, 5.23, 6.30,
          7.51, 8.88, 10.5, 12.5, 14.9],
    # Page 25.
    400: [0.21, 0.92, 1.99, 3.33, 4.94, 6.84, 9.00, 11.4, 14.2, 17.3,
          20.9, 24.9, 29.5, 35.4, 41.8],
    1000: [0.99, 3.67, 7.63, 12.7, 18.9, 26.4, 35.4, 46.0, 58.4, 73.0, 90.1],
    # Page 26.
    2500: [4.10, 14.9, 30.7, 52.0, 79.1, 113, 156, 209, 274, 353],
}


def m270_text() -> str:
    entry = resources.files("polecraft.data").joinpath("materials", "M270-35A.yaml")
    return entry.read_text(encoding="utf-8")


def m270_file() -> dict:
    return yaml.safe_load(m270_text())


def polarisation_at(curve: BHCurve, h: float) -> float:
    """J at field strength ``h`` in A/m, by inverting H(B) on a fine grid."""
    b = np.linspace(0.0, 2.5, 250001)
    return float(np.interp(h, curve.field_strength(b), b) - MU0 * h)


def test_m270_file_records_source_method_and_status() -> None:
    data = m270_file()
    assert "Cogent" in data["source"] and "2002-11" in data["source"]
    assert "IEC 60404-2" in data["source"]
    assert "not guaranteed" in " ".join(data["source"].lower().split())
    assert "copied from the brochure on 2026-11-23" in m270_text().lower()


def test_m270_file_cites_the_pages_its_values_are_on() -> None:
    # Page 5 is the table of guaranteed values; the typical ones are on 6.
    # The loss tables run to page 26 (2500 Hz).
    text, source = m270_text(), m270_file()["source"]
    assert "page 5" not in text
    assert "page 6" in source and "page 8" in source and "pages 18-26" in source


def test_m270_table_is_the_brochure_rows_exactly() -> None:
    rows = [tuple(row) for row in m270_file()["jh_curve"]]
    table = [(round(0.1 * k, 1), h) for k, h in enumerate(M270_H, start=1)]
    assert len(table) == 18
    assert rows == table


def test_m270_loss_rows_are_the_brochure_rows_exactly() -> None:
    data = m270_file()
    assert data["loss_W_per_kg"] == M270_LOSS
    assert "loss_step_T" not in data  # so the steps are the brochure's 0.1 T
    tables = load_core_loss("M270-35A").tables
    assert sorted(tables) == sorted(float(f) for f in M270_LOSS)
    for frequency, losses in M270_LOSS.items():
        j, p = tables[float(frequency)]
        assert np.array_equal(p, losses)
        assert j == pytest.approx(0.1 * np.arange(1, len(losses) + 1))


def test_m270_curve_is_the_file_table_with_j_turned_into_b() -> None:
    j, h = np.array(m270_file()["jh_curve"], dtype=float).T
    curve = load_material("M270-35A")
    assert curve.b[0] == 0.0 and curve.h[0] == 0.0
    assert np.array_equal(curve.h[1:], h)
    assert curve.b[1:] == pytest.approx(j + MU0 * h, rel=1e-15)
    assert curve.field_strength(curve.b[1:]) == pytest.approx(h, rel=1e-12)
    assert np.all(np.diff(curve.b) > 0.0) and np.all(np.diff(curve.b - MU0 * curve.h) > 0.0)


@pytest.mark.parametrize(
    ("h", "tolerance"),
    [
        # Page 6 is rounded to 0.01 T, so 0.005 T is as close as it can be
        # held; the curve is 0.003 T over at 2500 A/m and 0.003 T under at
        # 10000 A/m.
        (2500.0, 0.005),
        (10000.0, 0.005),
        # 5000 A/m falls in the knee between the 1.6 and 1.7 T rows, where
        # rows 0.1 T apart leave the curve 0.012 T under the page 6 value.
        # The value is not made a row: it would put a corner in the curve.
        (5000.0, 0.015),
    ],
)
def test_m270_curve_matches_the_brochure_typical_properties(h: float, tolerance: float) -> None:
    curve = load_material("M270-35A")
    assert polarisation_at(curve, h) == pytest.approx(M270_TYPICAL_J[h], abs=tolerance)


def test_m270_file_says_how_far_the_curve_is_from_the_typical_value_at_5000() -> None:
    text = " ".join(line.lstrip("# ") for line in m270_text().splitlines())
    quoted = re.search(
        r"At 5000 A/m it gives J = (\d\.\d{3}) T, (0\.\d{3}) T below the 1\.65 T", text
    )
    at_5000 = polarisation_at(load_material("M270-35A"), 5000.0)
    assert at_5000 == pytest.approx(float(quoted.group(1)), abs=5e-4)
    assert M270_TYPICAL_J[5000.0] - at_5000 == pytest.approx(float(quoted.group(2)), abs=5e-4)
    assert "rounded to 0.01 T" in text and "0.1 T apart" in text


def test_m270_curve_reaches_air_slope_beyond_1_8_t() -> None:
    curve = load_material("M270-35A")
    b = np.array([1.9, 2.0, 2.5, 4.0])
    slopes = MU0 * slope_of(curve, b)
    assert np.all(np.diff(slopes) > 0.0) and slopes[-1] == pytest.approx(1.0, abs=1e-3)
    assert 1.8 < curve.saturation < 2.0


@pytest.mark.parametrize("source", ["file", "docstring"])
def test_m270_tail_is_quoted_as_the_curve_gives_it(source: str) -> None:
    # The brochure has nothing above 1.8 T, so the file has no saturation
    # value and says in numbers what the default tail does instead.
    assert "saturation_polarisation_T" not in m270_file()
    text = m270_text() if source == "file" else BHCurve.__doc__
    curve = load_material("M270-35A")
    quoted = re.findall(r"(\d\.\d{3}) T at (\d+) kA/m", text)
    assert [int(field) for _, field in quoted] == [18, 30, 50, 100]
    for j, field in quoted:
        assert polarisation_at(curve, 1e3 * int(field)) == pytest.approx(float(j), abs=5e-4)
    level = re.search(r"tending to (\d\.\d{3}) T", text)
    assert curve.saturation == pytest.approx(float(level.group(1)), abs=5e-4)
    assert "likely" in text and "below the real" in " ".join(text.split())


def test_m270_loss_tables_load() -> None:
    data = load_core_loss("M270-35A")
    assert sorted(data.tables) == [50.0, 100.0, 200.0, 400.0, 1000.0, 2500.0]
    j, p = data.tables[50.0]
    assert j == pytest.approx(0.1 * np.arange(1, 19))
    # Page 8: 0.35 mm sheet, 7.65 kg/dm3, 52 micro-ohm cm.
    assert data.thickness == pytest.approx(0.35e-3)
    assert data.density == 7650.0 and data.resistivity == pytest.approx(52e-8)


def test_m270_losses_match_the_brochure_typical_properties() -> None:
    # Page 6 gives 1.01 W/kg at 1.0 T and 2.47 W/kg at 1.5 T, 50 Hz.
    j, p = load_core_loss("M270-35A").tables[50.0]
    assert p[np.isclose(j, 1.0)] == pytest.approx([1.01])
    assert p[np.isclose(j, 1.5)] == pytest.approx([2.47])


def test_eddy_coefficient_is_the_classical_sheet_formula() -> None:
    # pi^2 (0.35e-3)^2 / (6 x 52e-8 x 7650), worked by hand: 5.0655e-5.
    assert load_core_loss("M270-35A").eddy_coefficient() == pytest.approx(5.0655e-5, rel=1e-4)


def test_fit_recovers_known_coefficients() -> None:
    known = CoreLossCoefficients(hysteresis=0.02, eddy=5e-5, excess=4e-4)
    thickness, density = 0.35e-3, 7650.0
    resistivity = np.pi**2 * thickness**2 / (6.0 * known.eddy * density)
    j = 0.1 * np.arange(1, 16)
    tables = {f: (j, known.loss(f, j)) for f in (50.0, 200.0, 1000.0)}
    fitted = CoreLossData(thickness, density, resistivity, tables).fit()
    assert fitted.eddy == pytest.approx(known.eddy, rel=1e-12)
    assert fitted.hysteresis == pytest.approx(known.hysteresis, rel=1e-9)
    assert fitted.excess == pytest.approx(known.excess, rel=1e-9)


def test_m270_fit_reproduces_the_loss_tables() -> None:
    data = load_core_loss("M270-35A")
    fit = data.fit()
    assert fit.hysteresis > 0.0 and fit.excess > 0.0
    f, j, p = data.points(1000.0, 0.3)
    error = fit.loss(f, j) / p - 1.0
    assert np.sqrt(np.mean(error**2)) < 0.10
    assert np.max(np.abs(error)) < 0.25


def test_loss_terms_scale_with_frequency_as_stated() -> None:
    hysteresis = CoreLossCoefficients(1.0, 0.0, 0.0).loss([50.0, 100.0], 1.2)
    eddy = CoreLossCoefficients(0.0, 1.0, 0.0).loss([50.0, 100.0], 1.2)
    excess = CoreLossCoefficients(0.0, 0.0, 1.0).loss([50.0, 100.0], 1.2)
    assert hysteresis[1] / hysteresis[0] == pytest.approx(2.0)
    assert eddy[1] / eddy[0] == pytest.approx(4.0)
    assert excess[1] / excess[0] == pytest.approx(2.0**1.5)
    assert CoreLossCoefficients(1.0, 1.0, 1.0).loss(400.0, 0.0) == 0.0


def test_fit_keeps_the_range_of_the_points_it_used() -> None:
    data = load_core_loss("M270-35A")
    fit = data.fit()
    assert fit.frequency_range == (50.0, 1000.0)
    assert fit.min_polarisation == pytest.approx(0.3)
    # The 50 Hz table runs to 1.8 T, the others stop lower.
    assert fit.max_polarisation == pytest.approx(1.8)
    frequencies, highest = zip(*fit.polarisation_limits)
    assert frequencies == (50.0, 100.0, 200.0, 400.0, 1000.0)
    assert highest == pytest.approx([0.1 * len(M270_LOSS[int(f)]) for f in frequencies])
    assert highest == pytest.approx([1.8, 1.5, 1.5, 1.5, 1.1])
    narrow = data.fit(max_frequency=400.0, min_polarisation=0.5)
    assert narrow.frequency_range == (50.0, 400.0)
    assert narrow.min_polarisation == pytest.approx(0.5)
    assert narrow.max_polarisation == pytest.approx(1.8)
    assert len(narrow.polarisation_limits) == 4


def test_m270_fit_is_the_least_squares_fit_of_the_brochure_rows() -> None:
    # The same fit worked here from the rows above: the range is carried
    # beside the coefficients and takes no part in them.
    chosen = [(f, k, p) for f, row in M270_LOSS.items() if f <= 1000 for k, p in enumerate(row, 1)]
    f, j, p = (np.array(column, dtype=float) for column in zip(*chosen))
    f, j, p = f[j >= 3], 0.1 * j[j >= 3], p[j >= 3]
    kc = np.pi**2 * 0.35e-3**2 / (6.0 * 52e-8 * 7650.0)
    basis = np.column_stack([f * j**2, (f * j) ** 1.5]) / p[:, None]
    (kh, ke), _ = scipy.optimize.nnls(basis, 1.0 - kc * (f * j) ** 2 / p)
    fit = load_core_loss("M270-35A").fit()
    assert fit.hysteresis == pytest.approx(kh, rel=1e-9) and kh > 0.0
    assert fit.excess == pytest.approx(ke, rel=1e-9) and ke > 0.0
    assert fit.eddy == pytest.approx(kc, rel=1e-12)


def test_loss_inside_the_fitted_range_gives_no_warning() -> None:
    data = load_core_loss("M270-35A")
    fit = data.fit()
    f, j, _ = data.points(1000.0, 0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit.loss(f, j)
        fit.loss(50.0, 0.3)
        fit.loss(50.0, 1.8)
        fit.loss(1000.0, 1.1)
        fit.loss([50.0, 400.0, 1000.0], [[0.3], [1.0], [1.1]])
        # Between 400 Hz (to 1.5 T) and 1000 Hz (to 1.1 T) the limit is on
        # the line between them: 1.3 T at 700 Hz.
        fit.loss(700.0, 1.3)


@pytest.mark.parametrize(
    ("frequency", "peak", "message"),
    [
        (2500.0, 1.0, "2500 Hz against a fit over 50 to 1000 Hz"),
        (10.0, 1.0, "frequencies from 10 to 10 Hz"),
        (400.0, 0.1, "flux densities down to 0.1 T against a fit from 0.3 T up"),
        ([50.0, 2500.0], [1.0, 0.2], "2500 Hz.*; flux densities down to 0.2 T"),
        # Above every fitted point.
        (50.0, 2.1, "a flux density of 2.1 T at 50 Hz against a fit up to 1.8 T there"),
        # Inside the overall range, above what was fitted at that frequency.
        (1000.0, 1.8, "a flux density of 1.8 T at 1000 Hz against a fit up to 1.1 T there"),
        (400.0, 1.6, "1.6 T at 400 Hz against a fit up to 1.5 T there"),
        (700.0, 1.35, "1.35 T at 700 Hz against a fit up to 1.3 T there"),
        ([50.0, 1000.0], [[1.7], [1.2]], "1.7 T at 1000 Hz against a fit up to 1.1 T"),
    ],
)
def test_loss_outside_the_fitted_range_warns_and_still_gives_the_value(
    frequency: float | list, peak: float | list, message: str
) -> None:
    fit = load_core_loss("M270-35A").fit()
    with pytest.warns(LossRangeWarning, match=message):
        loss = fit.loss(frequency, peak)
    by_hand = CoreLossCoefficients(fit.hysteresis, fit.eddy, fit.excess).loss(frequency, peak)
    assert np.array_equal(loss, by_hand) and np.all(by_hand > 0.0)


def test_coefficients_given_by_hand_have_no_range_and_never_warn() -> None:
    coefficients = CoreLossCoefficients(0.02, 5e-5, 4e-4)
    assert coefficients.frequency_range is None and coefficients.min_polarisation is None
    assert coefficients.max_polarisation is None and coefficients.polarisation_limits is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coefficients.loss(1e5, 0.0)


@pytest.mark.parametrize("name", ["generic_35A_placeholder", "copper", "generic_NdFeB_placeholder"])
def test_material_without_loss_data_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="no loss data"):
        load_core_loss(name)


@pytest.mark.parametrize(
    ("losses", "message"),
    [
        ("{50: [0.1, 0.05]}", "must be positive and rise"),
        ("{50: [0.0, 0.05]}", "must be positive and rise"),
        ("{50: []}", "must map frequencies"),
        ("{-50: [0.1, 0.2]}", "must map frequencies"),
        ("[0.1, 0.2]", "must map frequencies"),
    ],
)
def test_bad_loss_table_is_rejected(
    losses: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = (
        "name: broken\ndescription: x\nsource: y\nkind: steel\n"
        "bh_curve: [[0, 0], [1, 100]]\nthickness_mm: 0.35\ndensity_kg_m3: 7650\n"
        f"resistivity_ohm_m: 5.2e-7\nloss_W_per_kg: {losses}\n"
    )
    (tmp_path / "materials").mkdir()
    (tmp_path / "materials" / "broken.yaml").write_text(text, encoding="utf-8")
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)
    with pytest.raises(ValueError, match=message):
        load_core_loss("broken")

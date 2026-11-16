"""Material model for the solver: reluctivity against flux density."""

import math
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import (
    AIR,
    COPPER,
    MU0,
    BHCurve,
    FrohlichSteel,
    LinearMaterial,
    Magnet,
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


def test_straight_line_table_is_a_linear_material() -> None:
    b = np.array([0.0, 0.4, 1.0, 1.7])
    curve = BHCurve(b, b / (1000.0 * MU0))
    sample = np.linspace(0.0, 3.0, 50)
    assert curve.reluctivity(sample) == pytest.approx(1.0 / (1000.0 * MU0), rel=1e-12)
    assert curve.reluctivity_derivative(sample) == pytest.approx(0.0, abs=1e-9 / MU0)


def test_beyond_the_table_the_curve_is_a_straight_line(steel: BHCurve) -> None:
    end_b, end_h = STEEL_B[-1], STEEL_H[-1]
    b = np.array([end_b + 0.1, end_b + 0.2, end_b + 0.4])
    h = steel.field_strength(b)
    slopes = (h - end_h) / (b - end_b)
    assert slopes == pytest.approx(slopes[0], rel=1e-12)
    # It leaves the last point no flatter than the last table interval.
    assert slopes[0] >= (STEEL_H[-1] - STEEL_H[-2]) / (STEEL_B[-1] - STEEL_B[-2])


def test_two_point_table() -> None:
    curve = BHCurve([0.0, 1.0], [0.0, 200.0])
    b = np.array([0.0, 0.3, 1.0, 2.0])
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
    ],
)
def test_bad_tables_are_rejected(b: list, h: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BHCurve(b, h)


def test_bad_table_message_names_the_row() -> None:
    with pytest.raises(ValueError, match="row 3"):
        BHCurve([0.0, 1.0, 0.9, 1.5], [0.0, 100.0, 200.0, 300.0])


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


def test_no_real_steel_grade_is_shipped_yet() -> None:
    names = material_names()
    assert names == ["generic_35A_placeholder"]


def test_unknown_material_is_rejected_with_the_known_names() -> None:
    with pytest.raises(ValueError, match="M270-35A.*generic_35A_placeholder"):
        load_material("M270-35A")


@pytest.mark.parametrize("name", ["../secrets", "a/b", "", "x.yaml"])
def test_material_names_cannot_be_paths(name: str) -> None:
    with pytest.raises(ValueError, match="material"):
        load_material(name)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("kind: magnet\nbh_curve: [[0, 0], [1, 100]]\n", "only 'kind: steel'"),
        ("- just\n- a list\n", "only 'kind: steel'"),
        ("kind: steel\n", "'bh_curve' must be a list"),
        ("kind: steel\nbh_curve: [[0, 0, 0], [1, 100, 5]]\n", "'bh_curve' must be a list"),
        ("kind: steel\nbh_curve: [[0, 0], [1, x]]\n", "'bh_curve' must be a list"),
        ("kind: steel\nbh_curve: [[0, 0], [1, 100], [0.9, 200]]\n", "flux density must increase"),
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

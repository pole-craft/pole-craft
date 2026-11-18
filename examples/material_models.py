"""The material models the solver uses: air, a magnet, and steel that saturates.

Run from any folder: python examples/material_models.py
Needs no mesher.

The solver works with reluctivity, H / B, the inverse of permeability. For
air and magnets it is constant. For steel it rises with the flux density as
the steel saturates, and comes from a BH table or from a formula.

The one steel in the library so far, generic_35A_placeholder, is a
placeholder: its table is worked out from a formula and is not data for any
real steel. The magnet below is a stand-in too.
"""

import numpy as np

from polecraft.core.materials import (
    AIR,
    MU0,
    FrohlichSteel,
    Magnet,
    load_material,
    material_names,
)


def main() -> None:
    print(f"materials in the library: {', '.join(material_names())}")

    # Linear materials: the same reluctivity at every flux density.
    magnet = Magnet(remanence=1.2, relative_permeability=1.05)
    print(f"\nair:    relative permeability {1.0 / (MU0 * float(AIR.reluctivity(1.0))):.2f}")
    print(
        f"magnet: remanence {magnet.remanence:g} T, relative permeability "
        f"{1.0 / (MU0 * float(magnet.reluctivity(1.0))):.2f} (a stand-in, not a grade)"
    )

    # The placeholder steel, read from its table, beside the formula the
    # table was made from: B = mu0 H + Js H / (a + H) with Js = 2 T, a = 300 A/m.
    table = load_material("generic_35A_placeholder")
    formula = FrohlichSteel(saturation=2.0, half_field=300.0)
    print("\nplaceholder steel (not real steel data)")
    print("   B        H from the table   H from the formula   relative permeability")
    for b in (0.5, 1.0, 1.5, 1.8, 2.0, 2.2):
        from_table = float(table.field_strength(b))
        from_formula = float(formula.field_strength(b))
        permeability = 1.0 / (MU0 * float(table.reluctivity(b)))
        print(
            f"{b:5.1f} T   {from_table:12.1f} A/m   {from_formula:14.1f} A/m   {permeability:14.0f}"
        )

    # Between the rows of the table the curve is interpolated, so that it
    # passes through every row and never turns back.
    b = np.linspace(0.0, 2.3, 500)
    rising = bool(np.all(np.diff(table.field_strength(b)) > 0.0))
    print(f"\nH rises with B all the way: {rising}")
    print(f"the table has {len(table.b)} rows, up to {table.b[-1]:.2f} T")


if __name__ == "__main__":
    main()

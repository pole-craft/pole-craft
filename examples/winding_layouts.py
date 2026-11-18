"""Winding layout, repeating slice and winding factor for several slot and pole counts.

Run from any folder: python examples/winding_layouts.py
Needs no mesher and no design file.

For a three-phase winding with a coil on every tooth. In the layout there is
one letter per tooth round the stator, giving the phase of the coil on it; a
lower-case letter is a coil wound the opposite way. Only the part that
repeats is shown.
"""

from polecraft.core.winding import (
    combination_problem,
    concentrated_winding,
    layout_string,
    symmetry,
    winding_factor,
)

COMBINATIONS = [(9, 8), (12, 8), (12, 10), (12, 14), (24, 28), (36, 42), (10, 8)]


def main() -> None:
    print("slots  poles   repeats   field in the next slice   winding factor   layout of one slice")
    for slots, poles in COMBINATIONS:
        problem = combination_problem(slots, poles)
        if problem:
            # Not every combination can carry a balanced three-phase winding.
            print(f"{slots:5d}  {poles:5d}   {problem}")
            continue
        repeat = symmetry(slots, poles)
        layout = layout_string(concentrated_winding(slots, poles))
        one_slice = layout[: repeat.slots]  # one letter per tooth
        next_slice = "reversed" if repeat.antiperiodic else "the same"
        print(
            f"{slots:5d}  {poles:5d}   {repeat.periods:4d} x    {next_slice:<22s}"
            f"   {winding_factor(slots, poles):10.4f}       {one_slice}"
        )


if __name__ == "__main__":
    main()

"""Pole-Craft: design and simulation tool for quasi-direct-drive (QDD) motors."""

__version__ = "0.0.1"

from polecraft.design import Design, design_from_dict, design_to_dict, load_design
from polecraft.validation import DesignError

__all__ = [
    "Design",
    "DesignError",
    "__version__",
    "design_from_dict",
    "design_to_dict",
    "load_design",
]

"""
This module tests diffraction_utils' io module.
"""

# Of course we need this while testing.
# pylint: disable=protected-access

from diffraction_utils.io import I07Nexus


def test_excalibur_name():
    """
    Make sure that we're spelling the detector names properly!
    """
    assert I07Nexus.excalibur_08_2023_roi.name == "excroi"
    assert I07Nexus.excalibur_04_2022.name == "exr"
    assert I07Nexus.pilatus_2021.name == "pil2roi"
    assert I07Nexus.pilatus_2022.name == "PILATUS"

"""
This module tests diffraction_utils' io module.
"""

# Of course we need this while testing.
# pylint: disable=protected-access

from diffraction_utils.io import (
    I07Nexus,
    detector_not_found,
    eiger_detector_info,
    excalibur_detector_info,
    p2m_detector_info,
    p100k_detector_info,
)


def test_excalibur_size():
    """
    check correct size finds excalibur detector type
    """
    assert I07Nexus.detector_size_dict[(515, 2069)] == excalibur_detector_info


def test_p2m_size():
    """
    check correct size finds p2m detector type
    """
    assert I07Nexus.detector_size_dict[(1679, 1475)] == p2m_detector_info


def test_p100k_size():
    """
    check correct size finds p100k detector type
    """
    assert I07Nexus.detector_size_dict[(195, 487)] == p100k_detector_info


def test_eiger_size():
    """
    check correct size finds eiger detector type
    """
    assert I07Nexus.detector_size_dict[(2162, 2068)] == eiger_detector_info


def test_no_detector_size():
    """
    check correct size finds no detector found
    """
    assert I07Nexus.detector_size_dict[(0, 0)] == detector_not_found

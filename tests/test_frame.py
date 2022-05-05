"""
This module contains a few simple tests for the Frame class.
"""

from diffraction_utils.frame_of_reference import Frame


def test_frame_names():
    """
    Explodes if someone accidentally changes the name of a frame.
    """
    assert Frame.lab == 'lab'
    assert Frame.sample_holder == 'sample holder'
    assert Frame.hkl == 'hkl'


def test_attr_names():
    """
    Explodes if any of the attributes of instances of Frame change names.
    """
    frame = Frame('')
    assert frame.frame_name == ''
    assert frame.diffractometer is None
    assert frame.scan_index is None

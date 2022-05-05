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


def test_frame_equality(rasor):
    """
    Make sure that our custom __eq__ is working.
    """
    frame1 = Frame('x', rasor, None)
    frame2 = Frame('x', rasor, None)

    assert frame1 == frame2

    frame3 = Frame('x', rasor, 3)
    assert frame3 != frame1

    frame4 = Frame('u', rasor, None)
    assert frame4 != frame1
    assert frame4 != frame3

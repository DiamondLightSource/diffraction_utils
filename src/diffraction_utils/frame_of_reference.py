"""
This module contains the Frame class, which is used to describe the frame of
reference in which vectors live.
"""

from .diffractometer_base import DiffractometerBase


class Frame:
    """
    Instances of this class contain enough information to uniquely identify a
    frame of reference. This isn't quite as simple as saying something like
    "sample frame", because the sample frame is generally a function of time
    during a scan. Instead, the frame of reference is completely described by
    a combination of an index and a string identifier.

    It is worth noting that in several special cases, scan_index does not need
    to be provided. For example, consider a crystal glued to a sample holder.
    In the Frame.sample_holder frame, descriptions of the crystal are
    independent of scan index. In that case, scan indices are only needed to
    transform to or from this frame: they are not needed to describe vectors in
    this frame.

    On the flip side, a vector describing a property of the crystal in the lab
    frame will need a scan_index and a diffractometer, since the diffractometer
    is generally moving during a scan.
    """

    sample_holder = 'sample holder'
    hkl = 'hkl'
    lab = 'lab'

    def __init__(self,
                 frame_name: str,
                 diffractometer: DiffractometerBase = None,
                 scan_index: int = None):
        self.frame_name = frame_name
        self.diffractometer = diffractometer
        self.scan_index = scan_index

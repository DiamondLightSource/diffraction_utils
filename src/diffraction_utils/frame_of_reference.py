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

    If it becomes necessary to generalise the concept of a frame in the future,
    this abstraction will simplify the process.
    """

    def __init__(self,
                 frame_name: str,
                 scan_index: int,
                 diffractometer: DiffractometerBase):
        self.frame_name = frame_name
        self.scan_index = scan_index
        self.diffractometer = diffractometer

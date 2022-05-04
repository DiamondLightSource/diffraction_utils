"""
This module contains the DiffractometerBase class.
"""

from .io import NexusBase


class DiffractometerBase:
    """
    This contains a generic description of what all diffractometers need to
    have.
    """

    def __init__(self, nexus: NexusBase) -> None:
        self.nexus = nexus

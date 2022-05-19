"""
The pytest conftest file.
"""

# Always necessary in conftest files.
# pylint: disable=redefined-outer-name

import os

from pytest import fixture

from diffraction_utils.io import I10Nexus, I07Nexus
from diffraction_utils.diffractometers.diamond_i07 import I07Diffractometer
from diffraction_utils.diffractometers.diamond_i10 import I10RasorDiffractometer


@fixture
def path_to_resources():
    """
    Returns the path to the resources file.
    """
    return "resources/" if os.path.isdir("resources/") else "tests/resources/"


@fixture
def path_to_2021_i07_nxs(path_to_resources: str):
    """
    Returns a path to a .nxs file acquired at beamline i07 in 2021.
    """
    return path_to_resources + "i07_2021.nxs"


@fixture
def path_to_04_2022_i07_nxs(path_to_resources: str):
    """
    Returns a path to a .nxs file acquired at beamline i07 in April 2022.
    """
    return path_to_resources + "i07_04_2022.nxs"


@fixture
def path_to_2022_i10_nxs(path_to_resources: str):
    """
    Returns a path to a .nxs file acquired at beamline i10 in 2022.
    """
    return path_to_resources + "i10_2022.nxs"


@fixture
def path_to_i10_data():
    """
    Returns the path to i10 data. This doesn't need to worry about paths for
    the diamond system, because _try_to_find_files will automagically find the
    file anyways.
    """
    return ('/Users/richard/Data/i10/07_04_22_Sam_Mapper_test/'
            '693862-pimte-files/')


@fixture
def rasor(path_to_2022_i10_nxs):
    """
    Returns an instance of I10RasorDiffractometer.
    """
    nexus = I10Nexus(path_to_2022_i10_nxs,
                     detector_distance=0.5)  # 50 cm detector distance.
    return I10RasorDiffractometer(nexus, [0, 1, 0],
                                  I10RasorDiffractometer.area_detector)


@fixture
def path_to_i07_soller_test_nxs():
    """
    Returns the path to the nexus file on my (richard.brearton@diamond.ac.uk)
    local machine. Tests using this fixture should only run on my machine.
    """
    return "/Users/richard/Data/i07/rsm_soller_test/418552/i07-418552.nxs"


@fixture
def soller_test_diffractometer(path_to_i07_soller_test_nxs):
    """
    Returns the I07Diffractometer instance corresponding to the data file
    stored in path_to_i07_soller_test_nxs.
    """
    return I07Diffractometer(path_to_i07_soller_test_nxs, [0, 1, 0],
                             I07Diffractometer.horizontal)


@fixture
def i07_04_2022_i07Nexus(path_to_04_2022_i07_nxs):
    """
    Returns the I07Nexus object corresponding to the nexus file stored in
    path_to_04_2022_i07_nxs.
    """
    return I07Nexus(path_to_04_2022_i07_nxs)


@fixture
def i07_04_2022_diffractometer(i07_04_2022_i07Nexus):
    """
    Returns the I07Diffractometer instance corresponding to the data file
    stored in path_to_04_2022_i07_nxs.
    """
    return I07Diffractometer(i07_04_2022_i07Nexus, [0, 1, 0],
                             I07Diffractometer.horizontal)

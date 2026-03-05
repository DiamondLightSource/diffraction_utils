"""
This module tests diffraction_utils' io module.
"""

# Of course we need this while testing.
# pylint: disable=protected-access


import numpy as np
from numpy.testing import assert_allclose

from diffraction_utils.io import I07Nexus



def test_attributes_i07_2021(path_to_2021_i07_nxs: str):
    """
    Make sure that i07 2021 .nxs files can be parsed and have valid attributes.
    """
    i07nexus = I07Nexus(path_to_2021_i07_nxs, locate_local_data=False)

    assert i07nexus.local_path == path_to_2021_i07_nxs
    assert i07nexus.probe_energy == 12.5e3
    assert i07nexus.nx_entry == i07nexus.nxfile['/entry/']
    assert i07nexus.nx_instrument == i07nexus.nxfile['/entry/instrument/']
    assert i07nexus.nx_detector == i07nexus.nxfile['/entry/instrument/excroi/']
    assert i07nexus.default_signal[0] == 342.7288888888887
    assert i07nexus.default_axis[0] == 0.010000170375764542
    assert i07nexus.default_signal_name == "Region_1_average"
    assert i07nexus.default_axis_name == "qdcd"
    assert i07nexus.default_nx_data_name == "excroi"
    assert i07nexus.default_nx_data.signal == "Region_1_average"
    assert i07nexus.scan_length == 51

    # Now check motor positions. This was a DCD scan, so none should change.
    assert_allclose(i07nexus.theta, -9.77)
    assert_allclose(i07nexus.alpha, 0.000553, atol=1e-6)
    assert_allclose(i07nexus.gamma, -9.77012)
    assert_allclose(i07nexus.delta, 0.04672)
    assert_allclose(i07nexus.omega, 30.001167)
    assert_allclose(i07nexus.chi, 3.640087e-06)


def test_excalibur_name():
    """
    Make sure that we're spelling the detector names properly!
    """
    assert I07Nexus.excalibur_detector_2021 == "excroi"
    assert I07Nexus.excalibur_04_2022 == "exr"
    assert I07Nexus.pilatus_2021 == "pil2roi"
    assert I07Nexus.pilatus_2022 == "PILATUS"


def test_attributes_i07_04_2022(path_to_04_2022_i07_nxs: str):
    """
    Make sure that .nxs files from i07 in April 2022 can be parsed.

    TODO: test against pixel size when this quantity is known.
    """
    i07nexus = I07Nexus(path_to_04_2022_i07_nxs, locate_local_data=False)

    assert i07nexus.local_path == path_to_04_2022_i07_nxs
    assert i07nexus.probe_energy == 19.9e3
    assert i07nexus.nx_entry == i07nexus.nxfile['/entry/']
    assert i07nexus.nx_instrument == i07nexus.nxfile['/entry/instrument/']
    assert i07nexus.nx_detector == i07nexus.nxfile['/entry/instrument/exr/']
    assert i07nexus.default_signal[0] == 1.0
    assert i07nexus.default_axis[0] == 0.900000841
    assert i07nexus.default_signal_name == "frameNo"  # ?????? ??????
    assert i07nexus.default_axis_name == "diff1delta"
    assert i07nexus.default_nx_data_name == "exr"
    assert i07nexus.default_nx_data.signal == "frameNo"  # ?????? ??????
    assert i07nexus.scan_length == 13
    assert i07nexus.image_shape == (515, 2069)

    delta_scan = np.arange(0.9, 1.4, 0.04)
    chi_scan = np.arange(0.45, 0.45 + 0.02 * len(delta_scan), 0.02)
    assert_allclose(i07nexus.theta, 0, atol=1e-6)
    assert_allclose(i07nexus.alpha, 0.001983, atol=1e-6)
    assert_allclose(i07nexus.gamma, -1.427e-05)
    assert_allclose(i07nexus.delta, delta_scan, atol=1e-4)
    assert_allclose(i07nexus.omega, 30)
    assert_allclose(i07nexus.chi, chi_scan, atol=1e-4)





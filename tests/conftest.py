"""
The pytest conftest file.
"""

import os

from pytest import fixture


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

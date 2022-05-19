"""
This module contains:

Parsing functions used to extract information from experimental files.

Classes used to help make parsing more modular. These include the NexusBase
class and its children.
"""

# We've gotta access the _value attribute on some NXobjects.
# pylint: disable=protected-access


import json
import os
from abc import abstractmethod
from typing import List, Dict, Tuple
from warnings import warn


import nexusformat.nexus.tree as nx
import numpy as np
from nexusformat.nexus import nxload
from PIL import Image as PILImageModule


from .region import Region
from .debug import debug
from .data_file import DataFileBase


class MissingMetadataWarning(UserWarning):
    """
    Warns a user that some metadata is missing.
    """


class NexusBase(DataFileBase):
    """
    This class contains *mostly* beamline agnostic nexus parsing convenience
    stuff. It's worth noting that this class still makes a series of assumptions
    about how data is laid out in a nexus file that can be broken. Instead of
    striving for some impossible perfection, this class is practical in its
    assumptions of how data is laid out in a .nxs file, and will raise if an
    assumption is violated. All instrument-specific assumptions that one must
    inevitably make to extract truly meaningful information from a nexus file
    are made in children of this class.

    Attrs:
        file_path:
            The local path to the file on the local filesystem.
        nxfile:
            The object produced by loading the file at file_path with nxload.
    """

    def __init__(self, local_path: str):
        super().__init__(local_path)
        self.nxfile = nxload(local_path)

    @property
    def src_path(self):
        """
        The name of this nexus file, as it was recorded when the nexus file was
        written.
        """
        return self.nxfile.file_name

    @property
    def detector(self):
        """
        Returns the NXdetector instance stored in this NexusFile.

        Raises:
            ValueError if more than one NXdetector is found.
        """
        det, = self.instrument.NXdetector
        return det

    @property
    def instrument(self):
        """
        Returns the NXinstrument instanced stored in this NexusFile.

        Raises:
            ValueError if more than one NXinstrument is found.
        """
        instrument, = self.entry.NXinstrument
        return instrument

    @property
    def entry(self) -> nx.NXentry:
        """
        Returns this nexusfile's entry.

        Raises:
            ValueError if more than one entry is found.
        """
        entry, = self.nxfile.NXentry
        return entry

    @property
    def default_signal(self) -> np.ndarray:
        """
        The numpy array of intensities pointed to by the signal attribute in the
        nexus file.
        """
        return self.default_nxdata[self.default_signal_name].nxdata

    @property
    def default_axis(self) -> np.ndarray:
        """
        Returns the nxdata associated with the default axis.
        """
        return self.default_nxdata[self.default_axis_name].nxdata

    @property
    def default_signal_name(self):
        """
        Returns the name of the default signal.
        """
        return self.default_nxdata.signal

    @property
    def default_axis_name(self) -> str:
        """
        Returns the name of the default axis.
        """
        return self.entry[self.entry.default].axes

    @property
    def default_nxdata_name(self):
        """
        Returns the name of the default nxdata.
        """
        return self.entry.default

    @property
    def default_nxdata(self) -> np.ndarray:
        """
        Returns the default NXdata.
        """
        return self.entry[self.default_nxdata_name]

    @abstractmethod
    def parse_raw_image_paths(self) -> List[str]:
        """
        Returns a list of paths to the .tiff images recorded during this scan.
        These are the same paths that were originally recorded during the scan,
        so will point at some directory in the diamond filesystem.
        """

    def get_local_image_paths(self, clue: str = '') -> List[str]:
        """
        Returns a list of local image paths. Raises FileNotFoundError if any of
        the images cannot be found. These local paths can be used to directly
        load the images.

        Args:
            clue (str):
                A hint as to where these images might be stored. A directory
                would make life easier. If this isn't given, this method will
                still search a large number of directories to try to find the
                images.

        Raises:
            FileNotFoundError if any of the images cant be found.
        """
        return _try_to_find_files(self.parse_raw_image_paths(),
                                  [clue, self.local_path])

    def load_image_arrays(self, clue: str = '', verbose=True) \
            -> List[np.ndarray]:
        """
        Tries to locate the images associated with this nexus file, if there are
        any. These images are stored as numpy arrays.

        Args:
            clue (str):
                A hint as to where these images might be stored. A directory
                would make life easier. If this isn't given, this method will
                still search a large number of directories to try to find the
                images.
        """
        # Do this in a loop as opposed to a comprehension so we can print.
        arrs = []
        for count, path in enumerate(self.get_local_image_paths(clue)):
            arrs.append(np.array(PILImageModule.open(path)))
            if verbose:
                print(f"Loading image number {count}.", end='\r', flush=True)
        return arrs


class I07Nexus(NexusBase):
    """
    This class extends NexusBase with methods useful for scraping information
    from nexus files produced at the I07 beamline at Diamond.
    """
    # Detectors.
    excalibur_detector_2021 = "excroi"
    excalibur_04_2022 = "exr"
    pilatus = "pil2roi"

    # Setups.
    horizontal = "horizontal"
    vertical = "vertical"
    dcd = "DCD"

    def __init__(self,
                 local_path: str,
                 detector_distance=None,
                 setup: str = 'horizontal',
                 diff_1=True):
        super().__init__(local_path)
        self.detector_distance = detector_distance
        if not diff_1:
            raise NotImplementedError(
                "Diffractometer 2 has not been implemented.")
        if setup == I07Nexus.dcd:
            raise NotImplementedError(
                "DCD nexus parsing has not been implemented.")
        if setup != I07Nexus.horizontal:
            raise NotImplementedError(
                "Only horizontal sample stage has been implemented.")

        # The nexusformat package is fragile, badly written and breaks in
        # parallel contexts. To get around this, some values are initialised.
        self.probe_energy = self.parse_probe_energy
        self.transmission = self.parse_transmission
        self.delta = self.parse_delta()
        self.gamma = self.parse_gamma()
        self.omega = self.parse_omega()
        self.theta = self.parse_theta()
        self.alpha = self.parse_alpha()
        self.chi = self.parse_chi()
        self.image_shape = self.parse_image_shape()
        self.pixel_size = self.parse_pixel_size()
        self.raw_image_paths = self.parse_raw_image_paths()

    def parse_delta(self) -> np.ndarray:
        """
        Returns a numpy array of the delta values throughout the scan.
        """
        return self._motors["diff1delta"]

    def parse_gamma(self) -> np.ndarray:
        """
        Returns a numpy array of the gamma values throughout the scan.
        """
        return self._motors["diff1gamma"]

    def parse_omega(self) -> np.ndarray:
        """
        Returns a numpy array of the omega values throughout the scan.
        """
        return self._motors["diff1omega"]

    def parse_theta(self) -> np.ndarray:
        """
        Returns a numpy array of the theta values throughout the scan.
        """
        return self._motors["diff1theta"]

    def parse_alpha(self) -> np.ndarray:
        """
        Returns a numpy array of the alpha values throughout the scan.
        """
        return self._motors["diff1alpha"]

    def parse_chi(self) -> np.ndarray:
        """
        Returns a numpy array of the chi values throughout the scan.
        """
        return self._motors["diff1chi"]

    @property
    def _motors(self) -> Dict[str, np.ndarray]:
        """
        A dictionary of all of the motor positions. This is only useful if you
        know some diffractometer specific keys, so it's kept private to
        encourage users to directly access the cleaner theta, two_theta etc.
        properties.
        """
        instr_motor_names = [
            "diff1delta", "diff1gamma", "diff1omega",
            "diff1theta", "diff1chi", "diff1alpha"]

        motors_dict = {
            x: np.ones(self.scan_length)*self.instrument[x].value._value
            for x in instr_motor_names}

        return motors_dict

    @property
    def local_data_path(self) -> str:
        """
        The local path to the data (.h5) file. Note that this isn't in the
        NexusBase class because it need not be reasonably expected to point at a
        .h5 file.

        Raises:
            FileNotFoundError if the data file cant be found.
        """
        file = _try_to_find_files(
            [self.parse_raw_image_paths()],
            [self.local_path])[0]
        return file

    @property
    def detector_name(self) -> str:
        """
        Returns the name of the detector that we're using. Because life sucks,
        this is a function of time.
        """
        if "excroi" in self.entry:
            return I07Nexus.excalibur_detector_2021
        if "exr" in self.entry:
            return I07Nexus.excalibur_04_2022
        if 'pil2roi' in self.entry:
            return I07Nexus.pilatus
        # Couldn't recognise the detector.
        raise NotImplementedError()

    def _get_ith_region(self, i: int):
        """
        Returns the ith region of interest found in the .nxs file.

        Args:
            i:
                The region of interest number to return. This number should
                match the ROI name as found in the .nxs file (generally not 0
                indexed).

        Returns:
            The ith region of interest found in the .nxs file.
        """
        x_1 = self.detector[self._get_region_bounds_key(i, 'x_1')][0]
        x_2 = self.detector[self._get_region_bounds_key(i, 'Width')][0] + x_1
        y_1 = self.detector[self._get_region_bounds_key(i, 'y_1')][0]
        y_2 = self.detector[self._get_region_bounds_key(i, 'Height')][0] + y_1
        return Region(x_1, x_2, y_1, y_2)

    @property
    def signal_regions(self) -> List[Region]:
        """
        Returns a list of region objects that define the location of the signal.
        Currently there is nothing better to do than assume that this is a list
        of length 1.
        """
        if self.detector_name == I07Nexus.excalibur_detector_2021:
            return [self._get_ith_region(i=1)]
        if self.detector_name == I07Nexus.excalibur_04_2022:
            # Make sure our code executes for bytes and strings.
            try:
                json_str = self.instrument[
                    "ex_rois/excalibur_ROIs"]._value.decode("utf-8")
            except AttributeError:
                json_str = self.instrument[
                    "ex_rois/excalibur_ROIs"]._value
            # This is badly formatted and cant be loaded by the json lib. We
            # need to make a series of modifications.
            json_str = json_str.replace('u', '')
            json_str = json_str.replace("'", '"')
            json_str = json_str.replace('x', '"x"')
            json_str = json_str.replace('y', '"y"')
            json_str = json_str.replace('width', '"width"')
            json_str = json_str.replace('height', '"height"')
            json_str = json_str.replace('angle', '"angle"')

            roi_dict = json.loads(json_str)
            return [Region.from_dict(roi_dict['Region_1'])]

        raise NotImplementedError()

    @property
    def background_regions(self) -> List[Region]:
        """
        Returns a list of region objects that define the location of background.
        Currently we just ignore the zeroth region and call the rest of them
        background regions.
        """
        if self.detector_name == I07Nexus.excalibur_detector_2021:
            return [self._get_ith_region(i)
                    for i in range(2, self._number_of_regions+1)]
        if self.detector_name == I07Nexus.excalibur_04_2022:
            # Make sure our code executes for bytes and strings.
            try:
                json_str = self.instrument[
                    "ex_rois/excalibur_ROIs"]._value.decode("utf-8")
            except AttributeError:
                json_str = self.instrument[
                    "ex_rois/excalibur_ROIs"]._value
            # This is badly formatted and cant be loaded by the json lib. We
            # need to make a series of modifications.
            json_str = json_str.replace('u', '')
            json_str = json_str.replace("'", '"')
            json_str = json_str.replace('x', '"x"')
            json_str = json_str.replace('y', '"y"')
            json_str = json_str.replace('width', '"width"')
            json_str = json_str.replace('height', '"height"')
            json_str = json_str.replace('angle', '"angle"')

            roi_dict = json.loads(json_str)
            bkg_roi_list = list(roi_dict.values())[1:]
            return [Region.from_dict(x) for x in bkg_roi_list]

        raise NotImplementedError()

    @property
    def parse_probe_energy(self):
        """
        Returns the energy of the probe particle parsed from this NexusFile.
        """
        return float(self.instrument.dcm1energy.value)*1e3

    @property
    def parse_transmission(self):
        """
        Proportional to the fraction of probe particles allowed by an attenuator
        to strike the sample.
        """
        return float(self.instrument.filterset.transmission)

    def parse_raw_image_paths(self):
        """
        Returns the raw path to the data file. This is useless if you aren't on
        site, but used to guess where you've stored the data file locally.
        """
        # This is far from ideal; there currently seems to be no standard way
        # to refer to point at information stored outside of the nexus file.
        # If you're a human, it's easy enough to find, but with code this is
        # a pretty rubbish task. Here I assume that data files are stored in
        # a specific location. This is probably fragile.

        if self.detector_name == I07Nexus.pilatus:
            path_array = self.detector["image_data"]._value
        if self.detector_name in [I07Nexus.excalibur_04_2022,
                                  I07Nexus.excalibur_detector_2021]:
            path_array = [self.instrument["excalibur_h5_data/exc_path"]._value]

        return [x.decode('utf-8') for x in path_array]

    @property
    def _region_keys(self) -> List[str]:
        """
        Parses all of the detector's dictionary keys and returns all keys
        relating to regions of interest.
        """
        return [key for key in self.detector.keys() if key.startswith("Region")]

    @property
    def _number_of_regions(self) -> int:
        """
        Returns the number of regions of interest described by this nexus file.
        This *assumes* that the region keys take the form f'region_{an_int}'.
        """
        split_keys = [key.split('_') for key in self._region_keys]

        return max([int(split_key[1]) for split_key in split_keys])

    def _get_region_bounds_key(self, region_no: int, kind: str) -> List[str]:
        """
        Returns the detector key relating to the bounds of the region of
        interest corresponding to region_no.

        Args:
            region_no:
                An integer corresponding the the particular region of interest
                we're interested in generating a key for.
            kind:
                The kind of region bounds keys we're interested in. This can
                take the values:
                    'x_1', 'width', 'y_1', 'height'
                where '1' can be replaced with 'start' and with/without caps on
                first letter of width/height.

        Raises:
            ValueError if 'kind' argument is not one of the above.

        Returns:
            A list of region bounds keys that is ordered by region number.
        """
        # Note that the x, y swapping is a quirk of the nexus standard, and is
        # related to which axis on the detector varies most rapidly in memory.
        if kind in ('x_1', 'x_start'):
            insert = 'X'
        elif kind in ('width', 'Width'):
            insert = 'Width'
        elif kind in ('y_1', 'y_start'):
            insert = 'Y'
        elif kind in ('height', 'Height'):
            insert = 'Height'
        else:
            raise ValueError("Didn't recognise 'kind' argument.")

        return f"Region_{region_no}_{insert}"

    def parse_pixel_size(self) -> float:
        """
        Returns the side length of pixels in the detector that's being used.
        """
        if self.detector_name in ['excroi', 'exr']:
            return 55e-6
        if self.detector_name in [I07Nexus.pilatus]:
            return 172e-6
        raise ValueError(f"Detector name {self.detector_name} is unknown.")

    def parse_image_shape(self) -> float:
        """
        Returns the shape of the images we expect to be recorded by this
        detector.
        """
        if self.detector_name in ['excroi', 'exr']:
            return 515, 2069
        if self.detector_name in [I07Nexus.pilatus]:
            return 1475, 1679
        raise ValueError(f"Detector name {self.detector_name} is unknown.")


class I10Nexus(NexusBase):
    """
    This class extends NexusBase with methods useful for scraping information
    from nexus files produced at the I10 beamline at Diamond.
    """

    # We might need to check which instrument we're using at some point.
    rasor_instrument = "rasor"

    def __init__(self, local_path: str, detector_distance: float = None):
        super().__init__(local_path)
        self.detector_distance = detector_distance

        # Warn the user if detector distance hasn't been set.
        if self.detector_distance is None:
            warn(MissingMetadataWarning(
                "Detector distance has not been set. At I10, sample-detector "
                "distance is not recorded in the nexus file, and must be "
                "input manually when using this library if it is needed."))

        # The nexusformat package is fragile, badly written and breaks in
        # parallel contexts. To get around this, some values are initialised.
        self.probe_energy = self.parse_probe_energy
        self.theta = self.parse_theta
        self.theta_area = self.parse_theta_area
        self.two_theta = self.parse_two_theta
        self.two_theta_area = self.parse_two_theta_area
        self.chi = self.parse_chi
        self.image_shape = self.parse_image_shape
        self.pixel_size = self.parse_pixel_size
        self.raw_image_paths = self.parse_raw_image_paths()

    @property
    def parse_probe_energy(self):
        """
        Returns the energy of the probe particle parsed from this NexusFile.
        """
        return float(self.instrument.pgm.energy)

    @property
    def _motors(self) -> Dict[str, np.ndarray]:
        """
        A dictionary of all of the motor positions. This is only useful if you
        know some diffractometer specific keys, so it's kept private to
        encourage users to directly access the cleaner theta, two_theta etc.
        properties.
        """
        instr_motor_names = ["th", "tth", "chi"]
        diff_motor_names = ["theta", "2_theta", "chi"]

        motors_dict = {
            x: np.ones(self.scan_length)*self.instrument.rasor.diff[y]._value
            for x, y in zip(instr_motor_names, diff_motor_names)}

        for name in instr_motor_names:
            try:
                motors_dict[name] = self.instrument[name].value._value
            except KeyError:
                pass
        return motors_dict

    @property
    def parse_theta(self) -> np.ndarray:
        """
        Returns the current theta value of the diffractometer, as parsed from
        the nexus file. Note that this will be different to thArea in GDA.
        """
        return self._motors["th"]

    @property
    def parse_two_theta(self) -> np.ndarray:
        """
        Returns the current two-theta value of the diffractometer, as parsed
        from the nexus file. Note that this will be different to tthArea in GDA.
        """
        return self._motors["tth"]

    @property
    def parse_theta_area(self) -> np.ndarray:
        """
        Returns the values of the thArea virtual motor during this scan.
        """
        return 180 - self.theta

    @property
    def parse_two_theta_area(self) -> np.ndarray:
        """
        Returns the values of the tthArea virtual motor during this scan.
        """
        return 90 - self.two_theta

    @property
    def parse_chi(self) -> np.ndarray:
        """
        Returns the current chi value of the diffractometer.
        """
        return 90 - self._motors["chi"]

    @property
    def parse_pixel_size(self) -> float:
        """
        All detectors on I10 have 13.5 micron pixels.
        """
        return 13.5e-6

    @property
    def parse_image_shape(self) -> Tuple[int]:
        """
        Returns the shape of detector images. This is easy in I10, since they're
        both 2048**2 square detectors.
        """
        return 2048, 2048

    def parse_raw_image_paths(self) -> List[str]:
        """
        Returns a list of paths to the .tiff images recorded during this scan.
        These are the same paths that were originally recorded during the scan,
        so will point at some directory in the diamond filesystem.
        """
        return [x.decode('utf-8') for x in self.default_signal]

    def load_image_array(self, index: int, clue: str = '') -> List[np.ndarray]:
        """
        Tries to locate an image associated with this nexus file. If the image
        is found, it is loaded as a numpy array and returned.

        Args:
            index (int):
                The index of the image array in the get_local_image_paths list.
            clue (str):
                A hint as to where these images might be stored. A directory
                would make life easier. If this isn't given, this method will
                still search a large number of directories to try to find the
                images.
        """
        path = self.get_local_image_paths(clue)[index]
        return np.array(PILImageModule.open(path))


def _try_to_find_files(filenames: List[str],
                       additional_search_paths: List[str]):
    """
    Check that data files exist if the file parsed by parser pointed to a
    separate file containing intensity information. If the intensity data
    file could not be found in its original location, check a series of
    probable locations for the data file. If the data file is found in one
    of these locations, update file's entry in self.data.

    Returns:
        :py:attr:`list` of :py:attr:`str`:
            List of the corrected, actual paths to the files.
    """
    found_files = []

    # If we had only one file, make a list out of it.
    if not hasattr(filenames, "__iter__"):
        filenames = [filenames]

    cwd = os.getcwd()
    start_dirs = [
        cwd,  # maybe file is stored near the current working dir
        # To search additional directories, add them in here manually.
    ]
    start_dirs.extend(additional_search_paths)

    local_start_directories = [x.replace('\\', '/') for x in start_dirs]
    num_start_directories = len(local_start_directories)

    # Now extend the additional search paths.
    for i in range(num_start_directories):
        search_path = local_start_directories[i]
        split_srch_path = search_path.split('/')
        for j in range(len(split_srch_path)):
            extra_path_list = split_srch_path[:-(j+1)]
            extra_path = '/'.join(extra_path_list)
            local_start_directories.append(extra_path)

    # This line allows for a loading bar to show as we check the file.
    for i, _ in enumerate(filenames):
        # Better to be safe... Note: windows is happy with / even though it
        # defaults to \
        filenames[i] = str(filenames[i]).replace('\\', '/')

        # Maybe we can see the file in its original storage location?
        if os.path.isfile(filenames[i]):
            found_files.append(filenames[i])
            continue

        # If not, maybe it's stored locally? If the file was stored at
        # location /a1/a2/.../aN/file originally, for a local directory LD,
        # check locations LD/aj/aj+1/.../aN for all j<N and all LD's of
        # interest. This algorithm is a generalization of Andrew McCluskey's
        # original approach.

        # now generate a list of all directories that we'd like to check
        candidate_paths = []
        split_file_path = str(filenames[i]).split('/')
        for j in range(len(split_file_path)):
            local_guess = '/'.join(split_file_path[j:])
            for start_dir in local_start_directories:
                candidate_paths.append(
                    os.path.join(start_dir, local_guess))

        # Iterate over each of the candidate paths to see if any of them contain
        # the data file we're looking for.
        found_file = False
        for candidate_path in candidate_paths:
            if os.path.isfile(candidate_path):
                # File found - add the correct file location to found_files
                found_files.append(candidate_path)
                found_file = not found_file
                debug.log("Data file found at " + candidate_path + ".")
                break

        # If we didn't find the file, tell the user.
        if not found_file:
            raise FileNotFoundError(
                "The data file with the name " + filenames[i] + " could "
                "not be found. The following paths were searched:\n" +
                "\n".join(candidate_paths)
            )
    return found_files

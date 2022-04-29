# diffraction_utils

A tool to help with parsing and extracting meaningful information from
diffraction data files (.nxs, .h5, .dat files etc). This library should handle
both parsing and running standard calculations against the parsed data.

B matrices are calculated from data files. If your favourite diffractometer's
data files aren't supported, raise an issue on the github repo. U matrices are
calculated at runtime from input arguments.

In the code, U matrices are referred to as "orientation matrices", while B
matrices are "goniometer matrices".

#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to download data.

They are imported as:

import OCDocker.Toolbox.Downloading as ocdown
'''

# Imports
###############################################################################
import os
import urllib.request

from tqdm import tqdm

import OCDocker.Toolbox.Printing as ocprint

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################
class DownloadProgressBar(tqdm):
    """Deal with the progress bar to track download. Extends the tqdm class."""


    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = 0) -> None:
        '''Update the progress bar.

        Parameters
        ----------
        b : int, optional
            Number of blocks transferred so far [1]
        bsize : int, optional
            Size of each block (in tqdm units) [1]
        tsize : int, optional
            Total size (in tqdm units). If [None] remains unchanged.

        Returns
        -------
        None
        '''

        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)






# Functions
###############################################################################
## Private ##

## Public ##
def download_url(url: str , out_path: str) -> None:
    '''Download a file from given url.

    Parameters
    ----------
    url : str
        The url to download the file from.
    out_path : str
        The path where the file will be downloaded.

    Returns
    -------
    None
    '''

    # Print verboosity
    ocprint.printv(f"Downloading a file from '{url}' and saving to {out_path}.")
    
    # Create the progress bar object
    with DownloadProgressBar(unit="B",
                             unit_scale=True,
                             miniters=1,
                             desc=url.split(os.path.sep)[-1]) as t:
        urllib.request.urlretrieve(url, filename=out_path, reporthook=t.update_to)
    return None

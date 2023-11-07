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
Authors: Rossi, A.D.; Torres, P.H.M.;
[The Federal University of Rio de Janeiro]
Contact info:
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
Av. Carlos Chagas Filho 373 - CCS - bloco G1-19,
Cidade Universitária - Rio de Janeiro, RJ, CEP: 21941-902
E-mail address: arturossi10@gmail.com
This project is licensed under Creative Commons license (CC-BY-4.0) (Ver qual)
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

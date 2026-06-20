#!/usr/bin/env python3

# Description
###############################################################################
'''
Download helpers with progress-bar support for files and datasets.

Usage:

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
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################
class DownloadProgressBar(tqdm):
    """``tqdm`` progress bar hooked to ``urllib`` download callbacks.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`tqdm.tqdm`.
    **kwargs
        Keyword arguments forwarded to :class:`tqdm.tqdm`.
    """


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

#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to prepare GEMDOCK files and run it.

TODO: Unfinished!!! http://gemdock.life.nctu.edu.tw/dock/

They are imported as:

import OCDocker.Docking.GEMDOCK as ocgemdock
'''

# Imports
###############################################################################
import errno
import json
import os

import numpy as np

from glob import glob
from typing import Dict, List, Tuple, Union

# No config needed - Future docking modules

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.IO as ocio
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun
import OCDocker.Toolbox.Validation as ocvalidation

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
class GEMDOCK:
    """ GEMDOCK object with methods for easy run. """


    def __init__(self, config_path: str, box_file: str, receptor: ocr.Receptor, prepared_receptor_path: str, ligand: ocl.Ligand, prepared_ligand_path: str, gemdock_log: str, output_gemdock: str, name: str = "", overwrite_config: bool = False, spacing: float = 2.9) -> None:
        '''Constructor of the class GEMDOCK.
        
        Parameters
        ----------
        configPath : str
            The path for the config file.
        boxFile : str
            The path for the box file.
        receptor : ocr.Receptor
            The receptor object.
        preparedReceptorPath : str
            The path for the prepared receptor.
        ligand : ocl.Ligand
            The ligand object.
        preparedLigandPath : str
            The path for the prepared ligand.
        gemdockLog : str
            The path for the GEMDOCK log file.
        outputGEMDOCK : str
            The path for the GEMDOCK output files.
        name : str, optional
            The name of the GEMDOCK object, by default "".
        spacing : float, optional
            The spacing between to expand the box, by default 2.9.

        Returns
        -------
        None
        '''

        pass
    

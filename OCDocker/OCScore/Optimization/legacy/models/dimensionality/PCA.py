#!/usr/bin/env python3

# Description
###############################################################################
''' Module with a helper to execute the Principal Component Analysis (PCA)
on the datasets.

It is imported as:

import OCDocker.OCScore.Optimization.PCA as ocpca
'''

# Imports
###############################################################################
import os
import pandas as pd

from sklearn.decomposition import PCA

import OCDocker.Error as ocerror
import OCDocker.OCScore.Utils.legacy.Data as ocscoredata
import OCDocker.OCScore.Utils.IO as ocscoreio
import OCDocker.Toolbox.Logging as oclogging

LOGGER = oclogging.get_logger("ocscore.dimensionality.pca")

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

## Public ##


def run_pca(
        df_path: str,
        variance: float,
        pca_path: str,
        verbose: bool = False
    ) -> str:
    ''' Function to run PCA on the datasets.

    Parameters
    ----------
    df_path : str
        The path to the DataFrame.
    variance : float
        The percentage of variance to be explained. (0.0 - 1.0)
    pca_path : str
        The path to save the PCA object. If empty, the current working
        directory is used.
    verbose : bool
        Whether to print the results.

    Returns
    -------
    str
        The path to the PCA object.

    Raises
    ------
    ValueError
        If the variance is not between 0 and 1.
    '''

    # Check if the variance is between 0 and 1
    if variance <= 0 or variance > 1:
        # User-facing error: invalid variance value
        ocerror.Error.value_error(f"The variance must be between 0 and 1. Got: {variance}")
        raise ValueError("The variance must be between 0 and 1.")

    # Convert the variance to string
    variance_str = str(variance * 100).replace('.0', '')

    # Resolve and lazily create output directory only when PCA is requested.
    requested_pca_path = str(pca_path or "").strip()
    output_dir = (
        os.path.abspath(os.path.expanduser(requested_pca_path))
        if requested_pca_path
        else os.getcwd()
    )
    try:
        os.makedirs(output_dir, exist_ok = True)
    except OSError as exc:
        ocerror.Error.create_dir(f"Could not create PCA output directory '{output_dir}': {exc}")
        raise

    # Define the path to save the PCA object
    pca_file_path = os.path.join(output_dir, f"pca{variance_str}.pkl")

    # Parse the data from the CSV files
    dudez_data, pdbbind_data, score_columns = ocscoredata.preprocess_df(df_path)

    # Create the PCA object
    pca = PCA(n_components = variance)

    # Perform PCA on the all datasets
    pdbbind_pca = pca.fit_transform(
        pdbbind_data.drop(
            columns = ['receptor', 'ligand', 'name', 'type', 'db', 'experimental'] + score_columns,
            errors = 'ignore'
        )
    )

    # Save the PCA object in pickle format (PDBbind only to be used later, since it is the dataset which will be used for the model)
    ocscoreio.save_object(pca, pca_file_path)

    if verbose:
        dudez_pca = pca.transform(
            dudez_data.drop(
                columns = ['receptor', 'ligand', 'name', 'type', 'db'] + score_columns,
                errors = 'ignore'
            )
        )

        # Create a DataFrame with the PCA results for each dataset then add the score columns back
        dudez_pca_df = pd.DataFrame(
            data = dudez_pca,
            columns = [f'PC{i+1}' for i in range(dudez_pca.shape[1])]
        )
        pdbbind_pca_df = pd.DataFrame(
            data = pdbbind_pca,
            columns = [f'PC{i+1}' for i in range(pdbbind_pca.shape[1])]
        )

        # Add the metadata columns back
        dudez_pca_df = pd.concat(
            [
                dudez_data[score_columns + ['receptor', 'ligand', 'name', 'type', 'db']],
                dudez_pca_df
            ],
            axis = 1
        )
        pdbbind_pca_df = pd.concat(
            [
                pdbbind_data[score_columns + ['receptor', 'ligand', 'name', 'type', 'db', 'experimental']],
                pdbbind_pca_df
            ],
            axis = 1
        )

        # Check for NaNs in the PCA datasets
        LOGGER.info("==== NaNs in PCA datasets ====")
        LOGGER.info("--------------------------------")
        LOGGER.info("DUDEz")
        LOGGER.info("%s", dudez_pca_df.isnull().sum())
        LOGGER.info("PDBbind")
        LOGGER.info("%s", pdbbind_pca_df.isnull().sum())

        # Compare the size of the datasets before and after PCA
        LOGGER.info("==== Dataset sizes ====")
        LOGGER.info("-----------------------")
        LOGGER.info("DUDEz")
        LOGGER.info("Before PCA: %d features", dudez_data.shape[1] - 5 - len(score_columns))
        LOGGER.info("After PCA scaling: %d features", dudez_pca_df.shape[1] - 5)
        LOGGER.info("PDBbind")
        LOGGER.info("Before PCA: %d features", pdbbind_data.shape[1] - 6 - len(score_columns))
        LOGGER.info("After PCA scaling: %d features", pdbbind_pca_df.shape[1] - 6)

    return pca_file_path

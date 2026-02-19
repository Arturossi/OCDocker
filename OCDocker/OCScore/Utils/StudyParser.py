#!/usr/bin/env python3

# Description
###############################################################################
'''
Parse Optuna study metadata and summarize results.

Usage:

import OCDocker.OCScore.Utils.StudyParser as ocstudy
'''

# Imports
###############################################################################

from typing import Any
import optuna

import pandas as pd

import OCDocker.Toolbox.Printing as ocprint

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

def analyze_studies(
        snames: list[str],
        storage: str,
        n_trials: int = 5,
        verbose: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    For each study, load trials, filter COMPLETE + dedupe,
    compute combined_metric = RMSE - AUC, then pull out
    top-n by RMSE (smallest), top-n by AUC (largest),
    and top-n by combined_metric (smallest).
    Ablation studies also get a 'features' column.
    Returns three DataFrames: df_rmse, df_auc, df_combined.
    '''

    rmse_results: list[dict[str, Any]] = []
    auc_results: list[dict[str, Any]] = []
    combined_results: list[dict[str, Any]] = []

    def _extend_results(
        top_df: pd.DataFrame,
        out_results: list[dict[str, Any]],
        *,
        include_combined_metric: bool = False,
    ) -> None:
        '''Convert top trials DataFrame to records and append to result list.
        
        Parameters
        ----------
        top_df : pd.DataFrame
            The DataFrame containing the top trials to be converted and appended.
        out_results : list[dict[str, Any]]
            The list to which the converted trial records will be appended.
        include_combined_metric : bool, optional
            Whether to include the combined_metric in the output records. Default is False.
        '''

        if top_df.empty:
            return

        trials = top_df["number"].astype(int).tolist()
        rmses = top_df["value"].tolist()
        aucs = top_df["user_attrs_AUC"].tolist()
        combined_metrics = top_df["combined_metric"].tolist() if include_combined_metric else None
        feature_masks = (
            top_df["user_attrs_Feature_Mask"].tolist()
            if is_ablation and "user_attrs_Feature_Mask" in top_df.columns
            else None
        )

        records: list[dict[str, Any]] = []
        for idx, (trial, rmse, auc) in enumerate(zip(trials, rmses, aucs)):
            entry = {
                "study_name": sname,
                "study_type": study_type,
                "trial": trial,
                "rmse": rmse,
                "auc": auc,
            }
            if combined_metrics is not None:
                entry["combined_metric"] = combined_metrics[idx]
            if feature_masks is not None:
                entry["features"] = feature_masks[idx]
            records.append(entry)

        out_results.extend(records)

    for sname in snames:
        if verbose:
            print(f"Loading {sname}")
        # skip unwanted studies
        if any(tag in sname for tag in ("AO", "LIG", "REC", "SF",
                                         "feature_selection", "Feature selection",
                                         "Pre_", "pre-")):
            continue

        try:
            study = optuna.load_study(study_name=sname, storage=storage)
        except Exception as e:
            print(f"Could not load {sname}: {e}")
            continue

        try:
            # Ask Optuna for only the fields used in this routine to reduce
            # dataframe size and processing overhead on large studies.
            df = study.trials_dataframe(
                attrs=("number", "value", "state", "user_attrs"),
                multi_index=False,
            )
        except TypeError:
            # Backward-compatibility for mocked/older Study objects.
            df = study.trials_dataframe()

        selected_cols = [
            col for col in ("number", "state", "value", "user_attrs_AUC", "user_attrs_Feature_Mask")
            if col in df.columns
        ]
        if selected_cols:
            df = df[selected_cols]
        df = df[df.state == "COMPLETE"].drop_duplicates(subset=["value", "user_attrs_AUC"])
        df["combined_metric"] = df.value - df.user_attrs_AUC
        df["number"] = df.number.astype(int)

        take = len(df) if (n_trials == -1 or n_trials > len(df)) else n_trials

        top_rmse     = df.nsmallest(take, "value")
        top_auc      = df.nlargest(take, "user_attrs_AUC")
        top_combined = df.nsmallest(take, "combined_metric")

        study_type = parse_study_type(sname, False, False, False)
        is_ablation = "Ablation" in sname

        _extend_results(top_rmse, rmse_results)
        _extend_results(top_auc, auc_results)
        _extend_results(top_combined, combined_results, include_combined_metric=True)

    df_rmse     = pd.DataFrame(rmse_results)
    df_auc      = pd.DataFrame(auc_results)
    df_combined = pd.DataFrame(combined_results)

    return df_rmse, df_auc, df_combined


def analyze_studies_old(
        snames : list[str],
        storage : str,
        n_trials : int = 5,
        verbose : bool = False
    ) -> pd.DataFrame:
    ''' Analyze the studies and get the n best trials. 
    
    Parameters
    ----------
    snames : list[str]
        The list of study names.
    storage : str
        The storage string for the database.
    n_trials : int
        The number of trials to get.
    verbose : bool
        Whether to print the results.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the results.
    '''

    # Create an empty list to store the results
    results = []

    # Define the previous studies flags
    autoencoder = False
    genetic_algorithm = False
    multiple_autoencoders = False

    # Iterate over the study names
    for sname in snames:
        if verbose:
            print(f"\nStudy: {sname}")

        if "AO" in sname:
            if "LIG" in sname or "REC" in sname or "SF" in sname:
                multiple_autoencoders = True
            else:
                autoencoder = True
            # Ignore the study (not needed)
            continue
        elif "feature_selection" in sname or "Feature selection" in sname:
            genetic_algorithm = True
            # Ignore the study (not needed)
            continue
        elif "Pre_" in sname or "pre-" in sname:
            # Ignore the study (not needed)
            continue

        try:
            # Load the study
            study = optuna.load_study(study_name = sname, storage = storage)
        except Exception as e:
            print(f"Error loading study {sname}: {e}")
            continue

        # Get the trials dataframe
        df = study.trials_dataframe()

        # Filter the trials that are complete
        df = df[df['state'] == 'COMPLETE']

        # Filter repeated trials (same value and user_attrs_AUC) (just in case)
        df = df.drop_duplicates(subset=['value', 'user_attrs_AUC'])

        # Calculate the combined metric
        df['combined_metric'] = df['value'] - df['user_attrs_AUC']

        # Convert the number to int
        df['number'] = df['number'].astype(int)

        # Sort the trials by RMSE, AUC and combined metric
        best_rmse_df = df.sort_values(by=['value'], ascending=[True])
        best_auc_df = df.sort_values(by=['user_attrs_AUC'], ascending=[False])
        best_df = df.sort_values(by=['combined_metric'], ascending=[True])

        # Get the study type from the name
        study_type = parse_study_type(sname, autoencoder, genetic_algorithm, multiple_autoencoders)

        autoencoder = False
        genetic_algorithm = False
        multiple_autoencoders = False

        # If n_trials are -1 or bigger than the len of the df, get all the trials
        if n_trials == -1 or n_trials > len(df):
            n_trials = len(df)

        # Get the n best trials
        for i in range(0, n_trials):
            # Append the results to the list
            result = {
                'study_name': sname,
                'study_type': study_type,
                'total_trials': len(df),
                'best_rmse_number': best_rmse_df['number'].iloc[i],
                'best_rmse_value': best_rmse_df['value'].iloc[i],
                'best_rmse_auc': best_rmse_df['user_attrs_AUC'].iloc[i],
            }

            if "Ablation" in sname:
                result['best_rmse_features'] = best_rmse_df['user_attrs_Feature_Mask'].iloc[i]

            result.update({
                'best_auc_number': best_auc_df['number'].iloc[i],
                'best_auc_value': best_auc_df['value'].iloc[i],
                'best_auc': best_auc_df['user_attrs_AUC'].iloc[i],
            })

            if "Ablation" in sname:
                result['best_auc_features'] = best_auc_df['user_attrs_Feature_Mask'].iloc[i]

            result.update({
                'best_combined_number': best_df['number'].iloc[i],
                'best_combined_metric': best_df['combined_metric'].iloc[i],
                'best_combined_value': best_df['value'].iloc[i],
                'best_combined_auc': best_df['user_attrs_AUC'].iloc[i]
            })

            if "Ablation" in sname:
                result['best_combined_features'] = best_df['user_attrs_Feature_Mask'].iloc[i]

            results.append(result)
                
            if verbose:
                ocprint.printv(f"{len(df)}\t{best_rmse_df['number'].iloc[i]}\t{best_rmse_df['value'].iloc[i]}\t{best_rmse_df['user_attrs_AUC'].iloc[i]}\t{best_auc_df['number'].iloc[i]}\t{best_auc_df['value'].iloc[i]}\t{best_auc_df['user_attrs_AUC'].iloc[i]}\t{best_df['number'].iloc[i]}\t{best_df['combined_metric'].iloc[i]}\t{best_df['user_attrs_AUC'].iloc[i]}")

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)

    # Return the DataFrame
    return results_df


def parse_study_type(
        name : str,
        autoencoder : bool = False,
        genetic_algorithm : bool = False,
        multiple_autoencoders : bool = False
    ) -> str:
    ''' Parse the study type from the study name.

    Parameters
    ----------
    name : str
        The name of the study.
    autoencoder : bool, optional
        Whether the study is an autoencoder study. Default is False.
    genetic_algorithm : bool, optional
        Whether the study is a genetic algorithm study. Default is False.
    multiple_autoencoders : bool, optional
        Whether the study is a multiple autoencoders study. Default is False.

    Returns
    -------
    str
        The study type.
    '''

    # Determine the dimensional method
    if autoencoder:
        dimensional = "AE"
    elif genetic_algorithm:
        dimensional = "GA"
    elif multiple_autoencoders:
        dimensional = "MAE"
    elif "PCA95" in name:
        dimensional = "PCA95"
    elif "PCA90" in name:
        dimensional = "PCA90"
    elif "PCA85" in name:
        dimensional = "PCA85"
    elif "PCA80" in name:
        dimensional = "PCA80"
    elif "ScoreOnly" in name:
        dimensional = "Scores Only"
    elif "NoScores" in name:
        dimensional = "No Scores"
    else:
        dimensional = ""

    # Determine the ML method
    if "XGB" in name or "XGBoost" in name:
        ml_method = "XGB"
    elif "NN" in name:
        ml_method = "NN"
    elif "Trans" in name:
        ml_method = "Transformer"
    else:
        ml_method = ""

    # Combine dimensional and ML method
    if dimensional:
        return f"{ml_method} + {dimensional}"
    else:
        return ml_method

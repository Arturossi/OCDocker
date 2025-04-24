#!/usr/bin/env python3

# Description
###############################################################################
'''
First set of primordial variables and functions that are used to initialise the
OCDocker library.\n

They are imported as:

import OCDocker.Toolbox.Constants as occ
'''

# Imports
###############################################################################

import math

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

# Functions
###############################################################################

def cal_to_J(cal: float) -> float:
    ''' Convert calories to Joules.

    Parameters
    ----------
    cal : float
        Value in calories.

    Returns
    -------
    float
        Value in Joules.
    '''

    return cal * 4.184

def J_to_cal(J: float) -> float:
    ''' Convert Joules to calories.

    Parameters
    ----------
    J : float
        Value in Joules.

    Returns
    -------
    float
        Value in calories.
    '''

    return J / 4.184

def C_to_K(C: float) -> float:
    ''' Convert Celsius to Kelvin.

    Parameters
    ----------
    C : float
        Value in Celsius.

    Returns
    -------
    float
        Value in Kelvin.
    '''

    return C + 273.15

def K_to_C(K: float) -> float:
    ''' Convert Kelvin to Celsius.

    Parameters
    ----------
    K : float
        Value in Kelvin.

    Returns
    -------
    float
        Value in Celsius.

    Raises
    ------
    ValueError
        If Kelvin is negative.
    '''

    # Check if K is negative
    if K < 0:
        raise ValueError("Kelvin cannot be negative.")

    return K - 273.15
  
def convert_Ki_Kd_to_dG(K: float, T: float = 298.15) -> float:
    ''' Convert equilibrium constant to Gibbs free energy.

    Parameters
    ----------
    K : float
        Equilibrium constant.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Gibbs free energy.
    '''

    # Calculate dG
    dG = R * T * math.log(K)
    
    return dG

def convert_dG_to_Ki_Kd(dG: float, T: float = 298.15) -> float:
    ''' Convert Gibbs free energy to equilibrium constant.

    Parameters
    ----------
    dG : float
        Gibbs free energy.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Equilibrium constant.
    '''

    # Calculate K
    K = math.exp(-dG / (R * T))
    
    return K
    

# Constants
###############################################################################

# Gas contant

# cal/(mol*K)
R = 1.9872036
# kcal/(mol*K)
Rk = 0.0019872036
# J/(mol*K)
RJ = 8.314462618
# KJ/(mol*K)
RJK = 0.008314462618

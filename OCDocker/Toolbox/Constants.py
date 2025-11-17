#!/usr/bin/env python3

"""
Constants and utility functions for unit conversions and thermodynamic calculations.

This module provides fundamental physical constants and conversion functions used
throughout the OCDocker library. It includes temperature constants, gas constants,
conversion factors, and utility functions for converting between different units
and calculating thermodynamic properties.

**Usage:**

.. code-block:: python

    import OCDocker.Toolbox.Constants as occ
    
    # Use constants
    temp_kelvin = occ.STANDARD_TEMPERATURE_K
    gas_constant = occ.RJ
    
    # Use conversion functions
    joules = occ.cal_to_J(100.0)  # Convert 100 calories to Joules
    kelvin = occ.C_to_K(25.0)  # Convert 25°C to Kelvin

**Constants:**

The module defines the following fundamental constants:

* **Temperature Constants:**
  * :const:`STANDARD_TEMPERATURE_K` - Standard temperature (298.15 K = 25°C)
  * :const:`ZERO_C_IN_K` - Absolute zero in Celsius (273.15 K = 0°C)

* **Conversion Constants:**
  * :const:`CAL_TO_J` - Calories to Joules conversion factor (4.184)

* **Gas Constants:**
  * :const:`R` - Gas constant in cal/(mol·K) (1.9872036)
  * :const:`Rk` - Gas constant in kcal/(mol·K) (0.0019872036)
  * :const:`RJ` - Gas constant in J/(mol·K) (8.314462618)
  * :const:`RJK` - Gas constant in kJ/(mol·K) (0.008314462618)

* **Unit Conversion:**
  * :const:`order` - Dictionary for unit conversion between different orders of magnitude

**Functions:**

The module provides conversion functions for:
* Energy units (calories ↔ Joules)
* Temperature units (Celsius ↔ Kelvin)
* Thermodynamic calculations (equilibrium constants ↔ Gibbs free energy)

.. note::
    All constants are defined at the module level and can be used in function
    default parameters. They are fundamental physical constants and should not
    be modified.
"""

# Imports
###############################################################################

import math

import OCDocker.Error as ocerror

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

# Constants (defined early for use in function defaults)
###############################################################################

# Temperature constants
STANDARD_TEMPERATURE_K: float = 298.15
"""Standard temperature in Kelvin (298.15 K = 25°C).

This is the standard temperature used in thermodynamic calculations.
Commonly used as a default value for temperature-dependent functions.

:type: float
"""

ZERO_C_IN_K: float = 273.15
"""Absolute zero in Celsius expressed in Kelvin (273.15 K = 0°C).

Used for converting between Celsius and Kelvin temperature scales.

:type: float
"""

# Conversion constants
CAL_TO_J: float = 4.184
"""Calories to Joules conversion factor.

This constant is used to convert energy values from calories to Joules.
The conversion factor is 4.184 J/cal.

:type: float
"""

# Gas constants
R: float = 1.9872036
"""Ideal gas constant in cal/(mol·K).

Gas constant used for thermodynamic calculations in calories per mole per Kelvin.
Value: 1.9872036 cal/(mol·K)

:type: float
"""

Rk: float = 0.0019872036
"""Ideal gas constant in kcal/(mol·K).

Gas constant used for thermodynamic calculations in kilocalories per mole per Kelvin.
Value: 0.0019872036 kcal/(mol·K)

:type: float
"""

RJ: float = 8.314462618
"""Ideal gas constant in J/(mol·K).

Gas constant used for thermodynamic calculations in Joules per mole per Kelvin.
This is the SI unit value: 8.314462618 J/(mol·K)

:type: float
"""

RJK: float = 0.008314462618
"""Ideal gas constant in kJ/(mol·K).

Gas constant used for thermodynamic calculations in kilojoules per mole per Kelvin.
Value: 0.008314462618 kJ/(mol·K)

:type: float
"""

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

    return cal * CAL_TO_J


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

    return J / CAL_TO_J


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

    return C + ZERO_C_IN_K


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
        # User-facing error: invalid temperature value
        ocerror.Error.value_error(f"Kelvin cannot be negative. Got: {K}") # type: ignore
        raise ValueError("Kelvin cannot be negative.")

    return K - ZERO_C_IN_K


def convert_Ki_Kd_to_dG(K: float, T: float = STANDARD_TEMPERATURE_K) -> float:
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


def convert_dG_to_Ki_Kd(dG: float, T: float = STANDARD_TEMPERATURE_K) -> float:
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

# Order dictionary for unit conversion
order: dict[str, dict[str, float]] = {
    "Y": {
        "Y": 1e0, "Z": 1e-3, "E": 1e-6, "P": 1e-9, "T": 1e-12, "G": 1e-15, "M": 1e-18, "k": 1e-21, "un": 1e-24, "c": 1e-26, "m": 1e-27, "u": 1e-30, "n": 1e-33, "p": 1e-36, "f": 1e-39, "a": 1e-42, "z": 1e-45, "y": 1e-48
    },
    "Z": {
        "Y": 1e3, "Z": 1e0, "E": 1e-3, "P": 1e-6, "T": 1e-9, "G": 1e-12, "M": 1e-15, "k": 1e-18, "un": 1e-21, "c": 1e-23, "m": 1e-24, "u": 1e-27, "n": 1e-30, "p": 1e-33, "f": 1e-36, "a": 1e-39, "z": 1e-42, "y": 1e-45
    },
    "E": {
        "Y": 1e6, "Z": 1e3, "E": 1e0, "P": 1e-3, "T": 1e-6, "G": 1e-9, "M": 1e-12, "k": 1e-15, "un": 1e-18, "c": 1e-20, "m": 1e-21, "u": 1e-24, "n": 1e-27, "p": 1e-30, "f": 1e-33, "a": 1e-36, "z": 1e-39, "y": 1e-42
    },
    "P": {
        "Y": 1e9, "Z": 1e6, "E": 1e3, "P": 1e0, "T": 1e-3, "G": 1e-6, "M": 1e-9, "k": 1e-12, "un": 1e-15, "c": 1e-17, "m": 1e-18, "u": 1e-21, "n": 1e-24, "p": 1e-27, "f": 1e-30, "a": 1e-33, "z": 1e-36, "y": 1e-39
    },
    "T": {
        "Y": 1e12, "Z": 1e9, "E": 1e6, "P": 1e3, "T": 1e0, "G": 1e-3, "M": 1e-6, "k": 1e-9, "un": 1e-12, "c": 1e-14, "m": 1e-15, "u": 1e-18, "n": 1e-21, "p": 1e-24, "f": 1e-27, "a": 1e-30, "z": 1e-33, "y": 1e-34
    },
    "G": {
        "Y": 1e15, "Z": 1e12, "E": 1e9, "P": 1e6, "T": 1e3, "G": 1e0, "M": 1e-3, "k": 1e-6, "un": 1e-9, "c": 1e-11, "m": 1e-12, "u": 1e-15, "n": 1e-18, "p": 1e-21, "f": 1e-24, "a": 1e-27, "z": 1e-30, "y": 1e-33
    },
    "M": {
        "Y": 1e18, "Z": 1e18, "E": 1e12, "P": 1e9, "T": 1e6, "G": 1e3, "M": 1e0, "k": 1e-3, "un": 1e-6, "c": 1e-8, "m": 1e-9, "u": 1e-12, "n": 1e-15, "p": 1e-18, "f": 1e-21, "a": 1e-24, "z": 1e-27, "y": 1e-30
    },
    "k": {
        "Y": 1e21, "Z": 1e18, "E": 1e15, "P": 1e12, "T": 1e9, "G": 1e6, "M": 1e3, "k": 1e0, "un": 1e-3, "c": 1e-5, "m": 1e-6, "u": 1e-9, "n": 1e-12, "p": 1e-15, "f": 1e-18, "a": 1e-21, "z": 1e-24, "y": 1e-27
    },
    "un": {
        "Y": 1e24, "Z": 1e21, "E": 1e18, "P": 1e15, "T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "un": 1e0, "c": 1e-2, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18, "z": 1e-21, "y": 1e-24
    },
    "c": {
        "Y": 1e26, "Z": 1e23, "E": 1e20, "P": 1e17, "T": 1e14, "G": 1e11, "M": 1e8, "k": 1e5, "un": 1e2, "c": 1e0, "m": 1e-1, "u": 1e-4, "n": 1e-7, "p": 1e-10, "f": 1e-13, "a": 1e-16, "z": 1e-19, "y": 1e-22
    },
    "m": {
        "Y": 1e27, "Z": 1e24, "E": 1e21, "P": 1e18, "T": 1e15, "G": 1e12, "M": 1e9, "k": 1e6, "un": 1e3, "c": 1e1, "m": 1e0, "u": 1e-3, "n": 1e-6, "p": 1e-9, "f": 1e-12, "a": 1e-15, "z": 1e-18, "y": 1e-21
    },
    "u": {
        "Y": 1e30, "Z": 1e27, "E": 1e24, "P": 1e21, "T": 1e18, "G": 1e15, "M": 1e12, "k": 1e9, "un": 1e6, "c": 1e4, "m": 1e3, "u": 1e0, "n": 1e-3, "p": 1e-6, "f": 1e-9, "a": 1e-12, "z": 1e-15, "y": 1e-18
    },
    "n": {
        "Y": 1e33, "Z": 1e30, "E": 1e27, "P": 1e24, "T": 1e21, "G": 1e18, "M": 1e15, "k": 1e12, "un": 1e9, "c": 1e7, "m": 1e6, "u": 1e3, "n": 1e0, "p": 1e-3, "f": 1e-6, "a": 1e-9, "z": 1e-12, "y": 1e-15
    },
    "p": {
        "Y": 1e36, "Z": 1e33, "E": 1e30, "P": 1e27, "T": 1e24, "G": 1e21, "M": 1e18, "k": 1e15, "un": 1e12, "c": 1e10, "m": 1e9, "u": 1e6, "n": 1e3, "p": 1e0, "f": 1e-3, "a": 1e-6, "z": 1e-9, "y": 1e-12
    },
    "f": {
        "Y": 1e39, "Z": 1e36, "E": 1e33, "P": 1e30, "T": 1e27, "G": 1e24, "M": 1e21, "k": 1e18, "un": 1e15, "c": 1e13, "m": 1e12, "u": 1e9, "n": 1e6, "p": 1e3, "f": 1e0, "a": 1e-3, "z": 1e-6, "y": 1e-9
    },
    "a": {
        "Y": 1e42, "Z": 1e39, "E": 1e36, "P": 1e33, "T": 1e30, "G": 1e27, "M": 1e24, "k": 1e21, "un": 1e18, "c": 1e16, "m": 1e15, "u": 1e12, "n": 1e9, "p": 1e6, "f": 1e3, "a": 1e0, "z": 1e-3, "y": 1e-6
    },
    "z": {
        "Y": 1e45, "Z": 1e42, "E": 1e39, "P": 1e36, "T": 1e33, "G": 1e30, "M": 1e27, "k": 1e24, "un": 1e21, "c": 1e19, "m": 1e18, "u": 1e15, "n": 1e12, "p": 1e9, "f": 1e6, "a": 1e3, "z": 1e0, "y": 1e-3
    },
    "y": {
        "Y": 1e48, "Z": 1e45, "E": 1e42, "P": 1e39, "T": 1e36, "G": 1e33, "M": 1e30, "k": 1e27, "un": 1e24, "c": 1e22, "m": 1e21, "u": 1e18, "n": 1e15, "p": 1e12, "f": 1e9, "a": 1e6, "z": 1e3, "y": 1e0
    }
}
"""Unit conversion dictionary for orders of magnitude.

This dictionary provides conversion factors between different orders of magnitude.
The structure is: ``order[source_unit][target_unit] = conversion_factor``

**Supported Units:**
    Y, Z, E, P, T, G, M, k, un, c, m, u, n, p, f, a, z, y

**Example Usage:**

.. code-block:: python

    import OCDocker.Toolbox.Constants as occ
    
    # Convert from nano (n) to micro (u)
    factor = occ.order["n"]["u"]  # Returns 1e-3
    
    # Convert from micro (u) to base unit (un)
    factor = occ.order["u"]["un"]  # Returns 1e-6

:type: dict[str, dict[str, float]]
"""

# Define __all__ for explicit public API
__all__ = [
    # Temperature constants
    "STANDARD_TEMPERATURE_K",
    "ZERO_C_IN_K",
    # Conversion constants
    "CAL_TO_J",
    # Gas constants
    "R",
    "Rk",
    "RJ",
    "RJK",
    # Unit conversion
    "order",
    # Conversion functions
    "cal_to_J",
    "J_to_cal",
    "C_to_K",
    "K_to_C",
    "convert_Ki_Kd_to_dG",
    "convert_dG_to_Ki_Kd",
]

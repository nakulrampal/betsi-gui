"""

"""
from pathlib import Path

import numpy as np
from scipy.interpolate import splrep, PchipInterpolator, pchip_interpolate


def get_data(input_file):
    """Read pressure and Nitrogen uptake data from file. Asserts pressures in units of bar.

    Args:
        input_file: Path the path to the input file

    Returns:
        Pressure and Quantity adsorbed.
    """
    input_file = Path(input_file)

    data = np.loadtxt(str(input_file), skiprows=1, delimiter=',')
    pressure = data[:, 0]
    q_adsorbed = data[:, 1]
    assert (pressure < 1.1).all(), "Relative pressure must lie between 0 and 1 bar."
    return pressure, q_adsorbed


def get_fitted_spline(pressure, q_adsorbed):
    """ Fits a cubic spline to the isotherm.

    Args:
        pressure: Array of relative pressure values.
        q_adsorbed: Array of Nitrogen uptake values.

    Returns:
        tck tuple of spline parameters.
    """
    return splrep(pressure, q_adsorbed, s=50, k=3, quiet=True)

def get_pchip_interpolation(pressure,q_adsorbed):
    """ Fits isotherm with shape preserving pchip interpolation
    
    Args:
        pressure: Array of relative pressure values
        q_adsorbed: Array of Nitrogen uptake values

    Returns:
        Pchip parameters
    """
    return PchipInterpolator(pressure,q_adsorbed, axis =0, extrapolate=None)

def isotherm_pchip_reconstruction(pressure, q_adsorbed):
    """ Fits isotherm with a pchip interpolation. Can use this to
    calculate BET area for difficult isotherms
    
    Args: pressure: Array of relative pressure values
    q_adsorbed: Array of Nitrogen uptake values
    
    Returns: Array of interpolated nitrogen uptake values
    
    """
    x_range = np.linspace(pressure[0], pressure[len(pressure) -1 ], 500)
    y_pchip = pchip_interpolate(pressure,q_adsorbed,x_range, der=0, axis=0)
    
    pressure = x_range
    q_adsorbed = y_pchip
        
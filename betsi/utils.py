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

    if str(input_file).find('.txt') != -1 or str(input_file).find('.aif') != -1:
        pressure, q_adsorbed = get_data_for_txt_aif(str(input_file))
    else:
        try:
            data = np.loadtxt(str(input_file), skiprows=0, delimiter=',')
            pressure = data[:, 0]
            q_adsorbed = data[:, 1]
        except ValueError:
            data = np.loadtxt(str(input_file), skiprows=1, delimiter=',')
            pressure = data[:, 0]
            q_adsorbed = data[:, 1]
    
    ## New lines added here
    comments_to_data = {'has_negative_pressure_points': False,\
                        'monotonically_increasing_pressure': True,\
                        'rel_pressure_between_0_and_1': True}
    ## removes negetive relative pressure points if any
    negative_pressure_indexes = np.where(pressure < 0)[0]
    if len(negative_pressure_indexes) > 0:
        comments_to_data['has_negative_pressure_points'] = True
        pressure = np.delete(pressure, negative_pressure_indexes)
        q_adsorbed = np.delete(q_adsorbed, negative_pressure_indexes)
    
    ## checks if relative pressure points are monotonically increasing, if not,
    ## removes problematic points
    if not (pressure == np.sort(pressure)).all():
        comments_to_data['monotonically_increasing_pressure'] = False
        temp_index = 0
        temp_pressure = pressure
        temp_q_adsorbed = q_adsorbed
        while temp_index < len(temp_pressure)-1:
            if temp_pressure[temp_index+1] <= temp_pressure[temp_index]:
                temp_pressure = np.delete(temp_pressure, [temp_index+1])
                temp_q_adsorbed = np.delete(temp_q_adsorbed, [temp_index+1])
            else:
                temp_index += 1
        pressure = temp_pressure
        q_adsorbed = temp_q_adsorbed
    
    ## checks if relative pressure points lie between 0 and 1 (bar)
    if not (pressure < 1.1).all():
        comments_to_data['rel_pressure_between_0_and_1'] = False
        pressure_above_one_indexes = np.where(pressure > 1.1)[0]
        pressure = np.delete(pressure, pressure_above_one_indexes)
        q_adsorbed = np.delete(q_adsorbed, pressure_above_one_indexes)
    
    comments_to_data['interpolated_points_added'] = False
    
    ## assert (pressure < 1.1).all(), "Relative pressure must lie between 0 and 1 bar."
    return pressure, q_adsorbed, comments_to_data


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
    ## x_range = np.linspace(pressure[0], pressure[len(pressure) -1 ], 500)
    ## y_pchip = pchip_interpolate(pressure,q_adsorbed,x_range, der=0, axis=0)
    
    ## pressure = x_range
    ## q_adsorbed = y_pchip
    
    ## New lines added from here
    ## Add new interpolated points using pchip interpolation while having the original data points in the list as well
    x_range = np.linspace(np.log10(pressure[0]), np.log10(pressure[len(pressure) -1 ]), len(pressure))
    delta_x = abs(x_range[1] - x_range[0])/2
    for p in np.log10(pressure[1:-1]):
        to_be_deleted_indexes = []
        index = np.searchsorted(x_range,p)
        if 0 <= index < len(x_range) and abs(p - x_range[index]) < delta_x:
            to_be_deleted_indexes.append(index)
        if 0 <= (index+1) < len(x_range) and abs(p - x_range[index+1]) < delta_x:
            to_be_deleted_indexes.append(index+1)
        if 0 <= (index-1) < len(x_range) and abs(p - x_range[index-1]) < delta_x:
            to_be_deleted_indexes.append(index-1)
        x_range = np.delete(x_range, to_be_deleted_indexes)
    x_range = np.append(x_range[1:-1], np.log10(pressure))
    x_range.sort()
    x_range = np.power(10, x_range)
    
    y_pchip = pchip_interpolate(pressure,q_adsorbed,x_range, der=0, axis=0)
    
    pressure_new = x_range
    q_adsorbed_new = y_pchip
    
    return pressure_new, q_adsorbed_new
        
def get_data_for_txt_aif(input_file):
    """ Read pressure and Nitrogen uptake data from file if the file extension is *.txt or *.aif.    
    this function will be called in the get_data() function, and should not be called alone anywhere in the code
    as it might cause some errors.
    
    Args:
        input_file: String of the path to the input file

    Returns:
        Pressure and Quantity adsorbed.
    """
    
    pressure = []
    q_adsorbed = []
    
    with open(input_file, 'r') as f:
        input_file_lines = f.readlines()
    
    if len(input_file_lines) == 0:
        pressure = np.array(pressure)
        q_adsorbed = np.array(q_adsorbed)
        return pressure, q_adsorbed
    
    index_start = 0
    index_stop = len(input_file_lines) - 1
    index_iter = 0
    first_loop_index = 0
    
    if input_file.find(".aif") != -1:
        for line in input_file_lines:
            if line.find("loop_") != -1 and first_loop_index == 0:
                first_loop_index = index_iter;
            if line.find("_adsorp_pressure") != -1:
                _adsorp_pressure_index = index_iter;
            if line.find("_adsorp_p0") != -1:
                _adsorp_p0_index = index_iter;
            if line.find("_adsorp_amount") != -1:
                index_start = index_iter
            if index_start > 0 and (line.find("loop_") != -1 or index_iter + 1 == len(input_file_lines)):
                index_stop = index_iter
                break
            index_iter += 1
        for index in range(index_start, index_stop + 1):
            val = []
            for t in input_file_lines[index].split():
                try:
                    val.append(float(t))
                except ValueError:
                    pass
            if len(val) >= 3:
                pressure.append(val[_adsorp_pressure_index - first_loop_index - 1]/val[_adsorp_p0_index - first_loop_index - 1])
                q_adsorbed.append(val[index_start - first_loop_index - 1])
    else:
        delimiter = " "
        if input_file_lines[round(len(input_file_lines)/2)].find(",") != -1:
            delimiter = ","
        elif input_file_lines[round(len(input_file_lines)/2)].find(" ") != -1:
            delimiter = " "
        elif input_file_lines[round(len(input_file_lines)/2)].find("\t") != -1:
            delimiter = "\t"
        for line in input_file_lines:
            val = []
            for t in line.split(delimiter):
                try:
                    val.append(float(t))
                except ValueError:
                    pass
            if len(val) >= 2:
                pressure.append(val[0])
                q_adsorbed.append(val[1])
    if len(pressure) == 0 or len(q_adsorbed) == 0:
        print("You must provide a valid input file!")  
        
    pressure = np.array(pressure)
    q_adsorbed = np.array(q_adsorbed)
    
    return pressure, q_adsorbed
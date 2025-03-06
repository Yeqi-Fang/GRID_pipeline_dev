import datetime
import json
import multiprocessing
from pathlib import Path

import numpy as np
import streamlit as st
import xlrd
from astropy.io import fits
from joblib import Parallel, delayed
from scipy.interpolate import KroghInterpolator
from scipy.optimize import curve_fit

import save as save

# Dictionary containing calibration file paths for different detectors
file = {'天宁01': Path('./DownloadData_TianNing-01/Calib'),
        '天宁02': Path('./DownloadData_TianNing-02/Calib')}


def EC_extract(file, index=0):
    """
    Extract energy calibration data from Excel file
    
    Parameters
    ----------
    file : str
        Path to the Excel file containing calibration data
    index : int, optional
        Sheet index in the Excel workbook (default: 0)
        
    Returns
    -------
    ECdata : ndarray
        Array containing extracted calibration data
    """
    workbook = xlrd.open_workbook(file)
    worksheet = workbook.sheet_by_index(index)
    rows = worksheet.nrows
    c = tuple(worksheet.row_values(i)[:] for i in range(1, rows))
    ECdata = np.array(c)
    return ECdata


def TDataTrans(Array, detector):
    '''
    This function is used to temperature correct the channel address
    Parameters
    ----------
    Array : three-dimensional matrix (math.) 4 X N X 4
        The first 4 represents four channels, N represents the number of data, and the second 4 contains [time, channel address, bias voltage, temperature] information.

    Returns
    -------
    address : three-dimensional matrix (math.) 4 X N X 4
        The first 4 represents four channels, N represents the number of data, and the second 4 contains [time, temperature-corrected channel address, bias voltage, temperature] information.
    '''

    # Load temperature calibration parameters from JSON file
    # at: quadratic coefficient for temperature correction
    # bt: linear coefficient for temperature correction
    # ct: constant term coefficient for temperature correction
    with open(file[detector] / 'TemCalib.json') as f:
        content = f.read()
        CalibPara = json.loads(content)

    at = CalibPara['at']
    bt = CalibPara['bt']
    ct = CalibPara['ct']

    address = [[[] for j in range(2)] for i in range(3)]

    # Apply temperature correction to channel address for each channel
    for chan in range(4):
        tk = Array[chan][:, 3]  # Get temperature values
        temp = at[chan] * (tk ** 2) + bt[chan] * tk + ct[chan]  # Calculate correction factor
        Array[chan][:, 1] = Array[chan][:, 1] / temp.T  # Apply correction
    address = Array
    return address


def E_pro(ECdata, chan, ERebParas, ESpaPara, ch):
    """
    Process energy calibration for a single channel value
    
    Parameters
    ----------
    ECdata : ndarray
        Energy calibration data
    chan : int
        Channel number (0-3)
    ERebParas : dict
        Energy rebuild parameters
    ESpaPara : dict
        Energy span parameters
    ch : float
        Channel address value to convert
        
    Returns
    -------
    float
        Calibrated energy value
    """
    # Extract calibration parameters
    a1 = ERebParas['a1']
    b1 = ERebParas['b1']
    c1 = ERebParas['c1']

    b2 = ERebParas['b2']
    c2 = ERebParas['c2']

    low = ESpaPara['low']
    middle = ESpaPara['middle']
    high = ESpaPara['high']

    # Apply different calibration formulas based on channel range
    if ch < low[chan]:
        return 0
    elif low[chan] <= ch < middle[chan]:
        # Quadratic calibration for low-middle range
        return a1[chan] * (ch ** 2) + b1[chan] * ch + c1[chan]
    elif middle[chan] <= ch <= high[chan]:
        # Linear calibration for middle-high range
        return b2[chan] * ch + c2[chan]
    elif ch > high[chan]:
        # Interpolation for values beyond high range
        channel = int(ch)
        # Use nearby points for interpolation
        C = ECdata[channel - 1:channel + 3][:, 0]
        E = ECdata[channel - 1:channel + 3][:, chan + 1]
        # Krogh interpolation
        KI = KroghInterpolator(C, E)
        return KI(ch)


def EnergyDataTrans(Array, detector):
    '''
    This function is used to energy scale a channel address.
    
    Parameters
    ----------
    Array : three-dimensional matrix (math.) 4X N X 4
        The first 4 represents four channels, N represents the number of data, and the second 4 contains [time, temperature-corrected channel address, bias voltage, temperature] information.

    Returns
    -------
    energy : three-dimensional matrix (math.) 4 X N X 4
        The first 4 represents four channels, N represents the number of data, and the second 4 contains [time, energy after channel address calibration, bias voltage, temperature] information
    '''

    # Utilize all available CPU cores for parallel processing
    num_cores = multiprocessing.cpu_count()

    # Load energy rebuild parameters from JSON file
    with open(file[detector] / 'ERebuild.json') as f:
        content = f.read()
        ERebParas = json.loads(content)

    # Load energy span parameters from JSON file
    with open(file[detector] / 'ESpan.json') as f:
        content = f.read()
        ESpaPara = json.loads(content)

    # Load calibration data from Excel file
    ECdata = EC_extract(file[detector] / 'EC_Interpolation.xls')
    energy = []

    # Process each channel in parallel
    for chan in range(4):
        results = np.array(Parallel(n_jobs=num_cores)(
            delayed(E_pro)(ECdata, chan, ERebParas, ESpaPara, ch) for ch in Array[chan][:, 1]))
        Array[chan][:, 1] = results.T
    
    # Extract time and energy data for each channel
    energy = [Array[i][:, 0:2] for i in range(4)]
    return energy


def f(x, a, b, c):
    """
    Gaussian function for curve fitting
    
    Parameters
    ----------
    x : array
        x values
    a : float
        Amplitude
    b : float
        Center/mean
    c : float
        Width/sigma
        
    Returns
    -------
    array
        Gaussian function values
    """
    return a * np.exp(-(x - b) ** 2 / c ** 2)


def fix59_5(Array):
    """
    Fix energy calibration to align with Am-241's 59.5 keV line
    
    Parameters
    ----------
    Array : list of arrays
        Energy data for each channel
        
    Returns
    -------
    Array : list of arrays
        Energy data with corrected energy values
    """
    for i in range(4):
        # Create histogram of energy values
        h = np.histogram(Array[i][:, 1], bins=np.arange(0, 1900, 0.1))
        x = h[1][150:1040]  # Extract x-values (bin edges)
        y = h[0][150:1040]  # Extract y-values (counts)
        
        # Find peak position
        ymax_index = np.argmax(y)
        c = np.array([y[ymax_index], x[ymax_index], 2])  # Initial guess for Gaussian fit
        
        # Fit Gaussian to the peak
        popt, pcov = curve_fit(f, x, y, c, maxfev=10000)
        
        # Shift all energies to align peak with 59.5 keV (Am-241)
        Array[i][:, 1] = Array[i][:, 1] + (59.5 - (popt[1] + 0.05)) * np.ones(np.size(Array[i][:, 1]))

    return Array


def fix(ExtractFile, detector):
    """
    Main function for processing data files - performs channel address calibration and energy reconstruction
    
    Parameters
    ----------
    ExtractFile : list
        List of file paths to process
    detector : str
        Detector name key for calibration files
        
    Returns
    -------
    FixFile : list
        List of paths to output fixed files
    """
    n = 0
    FixFile = []

    with st.spinner('正在进行道址校准与能量重建...'):  # Displaying spinner during processing
        for file_path in ExtractFile:
            # Read intermediate data
            Array = save.ReadMid(file_path)
            
            # Apply temperature correction to channel address
            Array = TDataTrans(Array, detector)
            
            # Convert corrected channel address to energy
            Array = EnergyDataTrans(Array, detector)
            
            # Optional: fix59_5 can be used to align with Am-241 line
            # Array = fix59_5(Array)

            # Create FITS file structure for saving results
            hdu = [save.HDUMaker(HDUtype='PrimaryHDU')]
            for i in range(4):
                hdu.append(save.HDUMaker(f'CH{i}_FIX_DATAS',
                                         [('Creator', 'SCUGRID', 'Sichuan University GRID team'),
                                          ('FileTime', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                                           'file created time')],
                                         [('UTC', Array[i][:, 0], 'D', 's'),
                                          ('Energy', Array[i][:, 1], 'E', 'keV'),
                                          ]))
            
            # Create output path and save FITS file
            fix_path = file_path.parents[1] / 'Fix' / file_path.relative_to(file_path.parent)
            save.FitsSaver(fix_path, hdu)
            
            # Track processed files
            FixFile.append(fix_path)
            n += 1
            
    st.success('道址校准与能量重建完成')  # Display success message when complete

    return FixFile

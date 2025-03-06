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

file = {'天宁01': Path('./DownloadData_TianNing-01/Calib'),
        '天宁02': Path('./DownloadData_TianNing-02/Calib')}


def EC_extract(file, index=0):
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

    # at为温度修正公式两次项系数
    # bt为温度修正公式一次项系数
    # ct为温度修正公式常数项系数

    with open(file[detector] / 'TemCalib.json') as f:
        content = f.read()
        CalibPara = json.loads(content)

    at = CalibPara['at']
    bt = CalibPara['bt']
    ct = CalibPara['ct']

    address = [[[] for j in range(2)] for i in range(3)]

    for chan in range(4):
        tk = Array[chan][:, 3]
        temp = at[chan] * (tk ** 2) + bt[chan] * tk + ct[chan]
        Array[chan][:, 1] = Array[chan][:, 1] / temp.T
    address = Array
    return address


def E_pro(ECdata, chan, ERebParas, ESpaPara, ch):
    a1 = ERebParas['a1']
    b1 = ERebParas['b1']
    c1 = ERebParas['c1']

    b2 = ERebParas['b2']
    c2 = ERebParas['c2']

    low = ESpaPara['low']
    middle = ESpaPara['middle']
    high = ESpaPara['high']

    if ch < low[chan]:
        return 0

    elif low[chan] <= ch < middle[chan]:
        return a1[chan] * (ch ** 2) + b1[chan] * ch + c1[chan]

    elif middle[chan] <= ch <= high[chan]:
        return b2[chan] * ch + c2[chan]

    elif ch > high[chan]:
        channel = int(ch)

        C = ECdata[channel - 1:channel + 3][:, 0]
        E = ECdata[channel - 1:channel + 3][:, chan + 1]

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

    num_cores = multiprocessing.cpu_count()

    with open(file[detector] / 'ERebuild.json') as f:
        content = f.read()
        ERebParas = json.loads(content)

    with open(file[detector] / 'ESpan.json') as f:
        content = f.read()
        ESpaPara = json.loads(content)

    ECdata = EC_extract(file[detector] / 'EC_Interpolation.xls')
    energy = []

    for chan in range(4):
        results = np.array(Parallel(n_jobs=num_cores)(
            delayed(E_pro)(ECdata, chan, ERebParas, ESpaPara, ch) for ch in Array[chan][:, 1]))
        Array[chan][:, 1] = results.T
    energy = [Array[i][:, 0:2] for i in range(4)]
    return energy


def f(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / c ** 2)


def fix59_5(Array):
    for i in range(4):
        h = np.histogram(Array[i][:, 1], bins=np.arange(0, 1900, 0.1))
        x = h[1][150:1040]
        y = h[0][150:1040]
        ymax_index = np.argmax(y)
        c = np.array([y[ymax_index], x[ymax_index], 2])
        popt, pcov = curve_fit(f, x, y, c, maxfev=10000)

        Array[i][:, 1] = Array[i][:, 1] + (59.5 - (popt[1] + 0.05)) * np.ones(np.size(Array[i][:, 1]))

    return Array


def fix(ExtractFile, detector):
    n = 0
    FixFile = []

    with st.spinner('正在进行道址校准与能量重建...'):
        for file_path in ExtractFile:

            Array = save.ReadMid(file_path)

            Array = TDataTrans(Array, detector)
            Array = EnergyDataTrans(Array, detector)
            # Array = fix59_5(Array)

            hdu = [save.HDUMaker(HDUtype='PrimaryHDU')]
            for i in range(4):
                hdu.append(save.HDUMaker(f'CH{i}_FIX_DATAS',
                                         [('Creator', 'SCUGRID', 'Sichuan University GRID team'),
                                          ('FileTime', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                                           'file created time')],
                                         [('UTC', Array[i][:, 0], 'D', 's'),
                                          ('Energy', Array[i][:, 1], 'E', 'keV'),
                                          ]))
            fix_path = file_path.parents[1] / 'Fix' / file_path.relative_to(file_path.parent)
            save.FitsSaver(fix_path, hdu)
            # np.save(file_path.split('.npy')[0]+r'_fix\\'+f'Fix_Observation_mission-{n}', Array)
            FixFile.append(fix_path)

            n += 1
    st.success('道址校准与能量重建完成')

    return FixFile

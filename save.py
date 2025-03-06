import datetime

import numpy as np
import pandas as pd
from astropy.io import fits


def HDUMaker(HDU_name='', header_list=[], coloums_list=[], HDUtype='BinTableHDU'):
    if HDUtype == 'PrimaryHDU':
        hdu = fits.PrimaryHDU()
    elif HDUtype == 'BinTableHDU':
        cols = []
        for col in coloums_list:
            try:
                cols.append(fits.Column(name=col[0], array=col[1], format=col[2], unit=col[3]))
            except:
                cols.append(fits.Column(name=col[0], array=col[1], format=col[2]))
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.verify()

        hdu.name = HDU_name
        for kvc in header_list:
            # key - value /comments
            hdu.header.append(fits.Card(*kvc))
    else:
        raise ValueError('HDUtype must be PrimaryHDU or BinTableHDU')
    return hdu


def FitsSaver(filename, hdu_list):
    if type(hdu_list) == list:
        hdu_list = fits.HDUList(hdu_list)
    hdu_list.writeto(filename, overwrite=True)
    hdu_list.close()


def ReadMid(file_path):
    fitsfile = fits.open(file_path)
    out = []
    for i in range(1, 5):
        try:
            out.append(np.vstack((fitsfile[i].data['UTC'],
                                  fitsfile[i].data['ADC'],
                                  fitsfile[i].data['Bias'],
                                  fitsfile[i].data['Tem'])).T)
        except:
            out.append(np.vstack((fitsfile[i].data['UTC'],
                                  fitsfile[i].data['Energy'],)).T)

    return out


def SaveScience(FixFile, detector):
    Ebfile = {
        '天宁01': './DownloadData_TianNing-01/Calib/Ebound.npy',
        '天宁02': './DownloadData_TianNing-02/Calib/Ebound.npy'}
    Eb = np.load(Ebfile[detector])
    bins = np.hstack((Eb[:, 1], Eb[-1, 2]))
    for file in FixFile:
        data = ReadMid(file)

        hdu1 = [HDUMaker(HDUtype='PrimaryHDU'),
                HDUMaker('EBOUND',
                         [('Creator', 'SCUGRID', 'Sichuan University GRID team'),
                          ('FileTime', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), 'file created time')],
                         [('CHANNEL', Eb[:, 0], 'I'),
                          ('E_MIN', Eb[:, 1], 'E', 'keV'),
                          ('E_MAX', Eb[:, 2], 'E', 'keV')
                          ])]

        # hdu2 = []#有效时间判断

        hdu3 = []
        times = []
        for i in range(4):
            times.append([data[i][:, 0].min(), data[i][:, 0].max()])
            data[i][:, 1] = np.array(pd.cut(data[i][:, 1], bins, labels=Eb[:, 0]))
            hdu3.append(HDUMaker(f'EVENTS{i}',
                                 [('Creator', 'SCUGRID', 'Sichuan University GRID team'),
                                  ('FileTime', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), 'file created time')],
                                 [('TIME', data[i][:, 0], 'D', 's'),
                                  ('PI', data[i][:, 1], 'I'),
                                  ('DEAD_TIME', 15 * np.ones((np.shape(data[i][:, 0])[0], 1)), 'B', 'ns'),
                                  ('EVT_TYPE', np.ones((np.shape(data[i][:, 0])[0], 1)), 'B')
                                  ]))
        times = np.array(times)
        hdu2 = [HDUMaker('GTI',
                         [('Creator', 'SCUGRID', 'Sichuan University GRID team'),
                          ('FileTime', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), 'file created time')],
                         [('START', np.array([times[:, 0].min()]), 'D', 's'),
                          ('STOP', np.array([times[:, 1].max()]), 'D', 's')
                          ])]
        # fits_path = file
        FitsSaver(file.parents[1] / 'Science' / file.relative_to(file.parent), hdu1 + hdu2 + hdu3)

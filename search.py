import os
import re
import struct

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

# Constants for data package structure and parsing
PACKAGE_LENGTH = 2048  # Length of each data package in bytes
PACKAGE_HEAD = b'\x6c\xe2\x6c\xe2.{2044}'  # Regular expression pattern for package header
# Dictionary defining byte indices for different fields within the buffer
BUFFER_INDEX = {'label': 4, 'iframe': 0, 'nframes': [10, 14], 'filetype': 14, 'ifile': 15,
                'nfileframes': [16, 20], 'ifileframe': [20, 24], 'length': [24, 26],
                'bcc': [2042, 2044], 'tail': [2044, 2048]}
# Dictionary mapping file type codes to human-readable names
FILETYPE = {0x01: "LOG", 0x02: "HK", 0x11: "SCI", 0x12: "TIME",
            0x13: "IV", 0x21: 'none'}  # Package type codes
DATA_START = 26  # Index where actual data content starts
DATA_LENGTH = 2016  # Length of data content in each package


def find_index(buffer, pattern):
    """
    Find all occurrences of a regex pattern in a binary buffer
    
    Parameters
    ----------
    buffer : bytes
        The binary data to search within
    pattern : bytes
        Regular expression pattern to search for

    Returns
    -------
    index : ndarray
        Array of tuples containing start and end indices of matched patterns
    """
    Pattern = re.compile(pattern, re.S)
    index = np.array([(ip.start(), ip.end()) for ip in Pattern.finditer(buffer)])
    return index


def subpackage(buffer, istarts, subfile, Type, nfiles):
    """
    Extract and save a subpackage from the buffer
    
    Parameters
    ----------
    buffer : bytes
        Full binary data
    istarts : list
        Start indices of data frames
    subfile : Path
        Output file path
    Type : str
        Type of subfile to extract
    nfiles : int
        File number to extract

    Returns
    -------
    int
        Status code (0 for success)
    """
    try:
        data = b''
        nfile = nfiles - 1
        for i in istarts:
            buf = buffer[i:i + PACKAGE_LENGTH]
            if int.from_bytes(buf[BUFFER_INDEX['ifileframe'][0]:BUFFER_INDEX['ifileframe'][1]],
                                'little') == 0 or i == 0:
                nfile += 1
            if FILETYPE[buf[BUFFER_INDEX['filetype']]] != Type or nfile != nfiles:
                break
            else:
                data += buf[DATA_START:DATA_START + DATA_LENGTH]
        with open(subfile, 'wb') as fout:
            fout.write(data)
    except:
        st.warning(f'包解析失败：{subfile}')
    return 0


def find_subfile(filePath):
    """
    Parse a DAT file to find and extract subfiles
    
    Parameters
    ----------
    filePath : Path
        Path to the DAT file

    Returns
    -------
    subfiles : list
        List of dictionaries containing information about extracted subfiles
    """
    istarts = []
    # if not os.path.exists(filePath.split('.dat')[0]):  # 判断文件是否存在
    #     os.mkdir(filePath.split('.dat')[0])
    
    dat_dir = filePath.parent / filePath.stem
    dat_dir.mkdir(exist_ok=True)

    with open(filePath, 'rb') as file:
        buffer = file.read()
    index = find_index(buffer, PACKAGE_HEAD)
    for istart, iend in index:
        if iend - istart == PACKAGE_LENGTH:  # 判断索引句段是否符合包长，及是否是包
            istarts.append(istart)

    subfiles = []
    nfiles = 0

    with st.spinner('正在拆分子文件...'):
        for i in range(len(istarts)):

            buf = buffer[istarts[i]:istarts[i] + PACKAGE_LENGTH]
            if int.from_bytes(buf[BUFFER_INDEX['ifileframe'][0]:BUFFER_INDEX['ifileframe'][1]], 'little') == 0 or \
                    istarts[i] == 0:  # 确保是子文件头而不是数据帧头
                try:
                    Type = FILETYPE[buf[BUFFER_INDEX['filetype']]]
                except:
                    Type = 'None'
                    st.warning('包解析错误')

                if Type == 'LOG':
                    subfile = filePath.parent / f'file-{nfiles}_{Type}.txt'  # 子包文件名
                else:
                    subfile = filePath.parent / f'file-{nfiles}_{Type}.bin'
                subfiles.append({'ifile': nfiles, 'filename': subfile, 'type': Type})

                if not subfile.exists():
                    subpackage(buffer, istarts[i:], subfile, Type, nfiles)  # 如果某子文件不存在，则对其分包

                nfiles += 1
    st.success('子文件拆分完成')
    return subfiles


def find_sci(buffer):
    """
    Find scientific data patterns in a binary buffer
    
    Parameters
    ----------
    buffer : bytes
        Binary data from a science file

    Returns
    -------
    feature : ndarray
        Array of index tuples for feature data
    """
    featureEventNum = 20
    LenFeature = featureEventNum * 24 + 8
    pattern_wf = b'\x5a\x5a\x99\x66\x99\x66\x5a\x5a.{' + bytes(f'{LenFeature}',
                                                               encoding='utf-8') + b'}\xaa\xaa\x99\x66\x99\x66\xaa\xaa'

    feature = find_index(buffer, pattern_wf)

    return feature


def cl02(dat_files):
    """
    Classify data files by their extensions
    
    Parameters
    ----------
    dat_files : list
        List of data file paths

    Returns
    -------
    cl : list
        Classified files
    Type : ndarray
        Array of file extensions
    """
    Type = []
    cl = []
    for dat_file in dat_files:
        if dat_file.suffix not in Type or Type == []:
            Type.append(dat_file.suffix)
            cl.append([])

    Type = np.array(Type)
    for dat_file in dat_files:
        cl[np.where(Type == dat_file.suffix)[0][0]].append(dat_file)

    return cl, Type


def find_subfile02(cl):
    """
    Alternative method to find subfiles for TianNing-02 data
    
    Parameters
    ----------
    cl : list
        List of file paths

    Returns
    -------
    subfiles : list
        List of dictionaries with file information
    """
    Filetype = {'0x1': {'0010': 'LOG', '0011': 'HK'},
                '0x2': {'0001': 'SCI', '0002': 'TIME', '0003': 'IV'}}

    subfiles = []
    for i in cl:
        subfile = {}
        print(i)
        b = re.compile(r'UDP207-(\d+).')
        num = int(b.findall(i.name)[0])
        file = hex(num)
        subfile['filename'] = i
        subfile['type'] = Filetype[file[0:3]][file[5:9]]
        try:
            subfile['ifile'] = int(file[3:5])
        except:
            subfile['ifile'] = file[3:5]
        subfiles.append(subfile)

    return subfiles


import csv
import ctypes
import datetime
import gc
import os
import re
import struct

import numpy as np
import streamlit as st
from scipy import interpolate
from tqdm import tqdm

import save as save
import search as fin


class Parser:
    """
    Data parser for GRID satellite data
    
    This class provides methods to extract and parse various types of data
    from the TianNing satellite data files.
    """
    
    def __init__(self, logger, csvfile, datfile, detector, filepath):
        """
        Initialize the Parser object
        
        Parameters
        ----------
        logger : Logger
            Logger object for recording parsing information
        csvfile : list
            List of CSV file paths containing telemetry data
        datfile : list
            List of data file paths containing science data
        detector : str
            Detector name ('天宁01' or '天宁02')
        filepath : Path
            Base file path for the observation data
        """
        self.logger = logger
        self.csvfile = csvfile
        self.logger.info(f'parser:init:csvfile:{csvfile}')
        self.datfile = datfile
        self.logger.info(f'parser:init:datfile:{datfile}')
        self.detector = detector
        self.filepath = filepath
        self.logger.info(f'parser:init:filepath:{filepath}')

    @staticmethod
    def Hough(time_data):
        """
        Apply Hough transform for time calibration
        
        Parameters
        ----------
        time_data : ndarray
            Time data to transform

        Returns
        -------
        K : ndarray
            K parameter matrix
        B : ndarray
            B parameter matrix
        V : ndarray
            Boolean index matrix
        """
        Feq = 1000000

        utc = time_data[:, 0]
        usc = time_data[:, 2]

        y0 = utc
        x0 = usc / Feq

        X = np.tile([x0], (len(x0), 1))
        V = (X == X.T)
        Y = np.tile([y0], (len(y0), 1))
        K = (Y - Y.T) / (X + V - X.T)
        B = Y - K * X

        return K, B, V

    @staticmethod
    def parser_sci(subfile):
        """
        Parse scientific data feature quantities
        
        Parameters
        ----------
        subfile : dict
            Dictionary containing scientific file information

        Returns
        -------
        ftinfo : list
            Scientific feature data including oscillator/f0, peak value, baseline
        """

        filename = subfile['filename']
        Frequence = 100000000
        featureEventNum = 20
        crcLib = ctypes.cdll.LoadLibrary('./crc16.so')

        with open(filename, 'rb') as f:
            buffer = f.read()

        feature = fin.find_sci(buffer)

        ftdata = []
        ftinfo = [[], [], [], []]  # 列表，每一个空内储存一个通道
        # all_count = []
        for i in feature:
            bias = 16
            decode = dict()
            curr_feature = buffer[i[0]: i[1]]
            decode['channel_n'] = struct.unpack(">H", curr_feature[8:10])[0]  # 通道 #struck解包时结果必然为一元组
            decode['time_stamp'] = []  # 时间
            decode['data_max'] = []  # 最大值数据
            decode['data_base'] = []  # 基线数据
            decode['crc'] = []
            buffer_crc = curr_feature[:bias]
            for _ in range(featureEventNum):
                decode['time_stamp'].append(struct.unpack('>Q', curr_feature[bias: bias + 8])[0])
                decode['data_max'].append(struct.unpack('>H', curr_feature[bias + 16: bias + 18])[0])
                decode['data_base'].append(struct.unpack('>H', curr_feature[bias + 18: bias + 20])[0])

                buffer_crc = buffer_crc + curr_feature[bias:bias + 20]
                crc = struct.unpack('>H', curr_feature[bias + 22:bias + 24])[0]
                decode['crc'].append(
                    crc == crcLib.crc16_XModem(ctypes.c_char_p(buffer_crc), ctypes.c_uint(len(buffer_crc))))
                bias += 24

            ftdata.append(decode)
        gc.disable()
        for v in tqdm(ftdata):
            ftinfo[v['channel_n']] += [[v['time_stamp'][i] / Frequence, v['data_max'][i] - v['data_base'][i]] for i in
                                       range(len(v['time_stamp'])) if v['crc'][i]]
            # all_count = all_count+[v['time_stamp'][i]/Frequence for i in range(len(v['time_stamp']))]
            # 将相应数据分类到各通道中
        gc.enable()
        ftinfo = [np.array(v) for v in ftinfo]
        return ftinfo

    def parser_time(self, subfile):
        """
        Parse time data for time axis reconstruction
        
        Parameters
        ----------
        subfile : dict
            Dictionary containing time file information

        Returns
        -------
        houghdata : dict
            Data after Hough transformation
        k_max : float
            k value for time reconstruction
        b_max : float
            b value for time reconstruction
        """
        TIME_HEAD = b'\x1a\xcf\xfc\x1d.{2}\x90.{245}\x2e\xe9\xc8\xfd'
        buf = b''

        filename = subfile['filename']

        with open(filename, 'rb') as f:
            buffer = f.read()

        index = fin.find_index(buffer, TIME_HEAD)
        for istart, iend in index:
            if iend - istart == 256:
                buf += buffer[istart + 11:istart + 251]

        time_data = np.array(struct.unpack('>' + 'LLQ' * (len(buf) // 16), buf))
        time_data = time_data.reshape((-1, 3))

        delete_index = []
        for i in range(len(time_data[:, 0]) - 1):
            if time_data[i, 0] == time_data[i + 1, 0] or time_data[i, 1] == time_data[i + 1, 1]:
                delete_index.append(i)

        time_data = np.delete(time_data, delete_index, axis=0)
        time_data = np.delete(time_data, len(time_data) - 1, axis=0)

        K, B, V = self.Hough(time_data)

        k_t = K[np.logical_not(V)].flatten()
        b_t = B[np.logical_not(V)].flatten()
        k = k_t[(k_t > 0.96) & (k_t < 1.04)]
        b = b_t[(k_t > 0.96) & (k_t < 1.04)]
        b_tm = np.median(b)
        k_new = k[(b > b_tm - 20) & (b < b_tm + 20)]
        b_new = b[(b > b_tm - 20) & (b < b_tm + 20)]

        h, xedges, yedges = np.histogram2d(k_new, b_new, [1000, 1000])
        loc_y, loc_x = np.where(h == h.max())

        k_max = xedges[loc_x[0]]
        b_max = yedges[loc_y[0]]

        houghdata = {'xedges': xedges, 'yedges': yedges, 'h': h}

        return houghdata, k_max, b_max

    def parser_HK(self, subfile):
        """
        Parse housekeeping (HK) files
        
        Parameters
        ----------
        subfile : dict
            Dictionary containing HK file information

        Returns
        -------
        HKdatas : ndarray
            Matrix with UTC time, 4 SiPM channels' voltage, current, temperature
        """
        HK_HEAD = b'\x4d\x3c.{1}\x1a.{51}'
        HKdatas = [[], [], [], []]

        filename = subfile['filename']
        with open(filename, 'rb') as f:
            buffer = f.read()

        index = fin.find_index(buffer, HK_HEAD)
        for i in index:
            data = buffer[i[0]:i[1]]
            if self.detector == '天宁01':
                UTC_time = struct.unpack('>L', data[4: 8])[0]
            else:
                UTC_time = struct.unpack('<L', data[4: 8])[0]
            SiPM_average_voltage = struct.unpack('<' + 'H' * 4, data[20: 28])
            # SiPM_average_current =  struct.unpack('<'+'H'*4, data[28: 36])
            SiPM_average_temperature = struct.unpack('<' + 'H' * 4, data[36: 44])
            for k in range(4):
                if self.detector == '天宁02':
                    HKdatas[k].append([UTC_time, SiPM_average_voltage[k] * 0.001,
                                       # SiPM_average_current[k],
                                       SiPM_average_temperature[k] / 256])
                else:
                    HKdatas[k].append([UTC_time, SiPM_average_voltage[k],
                                       # SiPM_average_current[k],
                                       SiPM_average_temperature[k] / 256])

        HKdatas = np.array(HKdatas)

        return HKdatas

    def parser_Telem(self, telemfile):
        """
        Parse telemetry package data
        
        Parameters
        ----------
        telemfile : Path
            Path to telemetry package file

        Returns
        -------
        telem : list
            List of telemetry data arrays for each channel
        """
        telem = []

        with open(telemfile, 'r') as f:
            for i in csv.reader(f, skipinitialspace=True):
                if i[1] == 'UTC时间' or i[1] == 'sipm平均电压' or i[1] == 'sipm平均温度':
                    telem.append(float(i[4]))

            telem = np.array(telem)
            if np.size(telem) % 3 != 0:
                telem = telem[0:np.size(telem) - np.size(telem) % 3]
            telem = telem.reshape((-1, 3))

        delete_index = []
        for i in range(len(telem)):
            if telem[i, 0] == 0. or telem[i, 0] > 8e8 or telem[i, 1] == 65.535 or telem[i, 2] == 255.99609375:
                delete_index.append(i)

        telem = np.delete(telem, delete_index, axis=0)
        self.logger.info(f'parser:parser_Telem:telem:{telem}')
        return [telem, telem, telem, telem]

    def parser_LOG(self, subfiles, Type=None):
        """
        Analyze log files and classify subfiles
        
        Parameters
        ----------
        Type : str or None
            Log type for sorting ('排序' or None)
        subfiles : list
            List of subfile dictionaries

        Returns
        -------
        subfiles_classify : list or ndarray
            Classified subfiles by observation task
        """

        if Type == '排序':
            subfiles_classify = []
            for i in subfiles:
                if i['type'] == 'TIME' or i['type'] == 'SCI':
                    subfiles_classify.append(i)
            try:
                subfiles_classify = np.array(subfiles_classify).reshape(-1, 2)
                return subfiles_classify
            except:
                return []
        else:

            def Sort(elem):
                return elem[0]

            # ！此处为分组规则
            if self.detector == '天宁01':
                _Pattern = (r'(?P<FileNo>file-[0-9]+); name: /starware/[a-zA-Z/]+/(?P<FileName>(?P<OrderNO>[0-9]+)_['
                            r'A-Za-z]+\.[a-z]+); (?P<FileDate>size: [0-9]+ Bytes)')
                group_pat = (r"file-(?P<FileNo>[0-9]+);[0-9]+_(?P<FileName>[A-Za-z]+)\.[a-z]+;(?P<OrderNo>[0-9]+);("
                             r"?P<FileDate>[a-zA-Z0-9: ]+)")
            elif self.detector == '天宁02':
                _Pattern = (r'infile : /starware/GridData/gamma/(?P<FileName>(?P<OrderNO>[0-9]+)_[A-Za-z]+\.[a-z]+), '
                            r'outfile: /starware/Download/(?P<FileNo>[0-9]+)_(?P<File>[0-9]+)\.bin')
                group_pat = r"[0-9]+_(?P<FileName>[A-Za-z]+)\.[a-z]+;(?P<OrderNo>[0-9]+);(?P<FileNo>[0-9]+)"
            buf = ''
            LogFileList = []

            for i in subfiles:
                if i['type'] == 'LOG':
                    LogFileList.append(i['filename'])

            LogFileList = np.array(LogFileList)

            for Logfile in LogFileList:
                with open(Logfile) as f:
                    buf = buf + f.read()

            MatchList = re.findall(_Pattern, buf)
            DataList = []
            for x in range(len(MatchList)):
                List1 = re.match(
                    group_pat, ";".join(MatchList[x]))
                if List1.group("FileName") == "observe" or List1.group("FileName") == "timeline":
                    DataList.append([List1.group("FileNo"), List1.group("OrderNo")])
            DataList.sort(key=Sort)

            if not DataList:
                return []

            DataList = np.array(DataList)
            tables = []
            table = []
            for i in range(len(DataList[:, 1]) - 1):
                try:
                    table.append(int(DataList[i, 0]))
                    if DataList[i, 1] != DataList[i + 1, 1]:
                        tables.append(table)
                        table = []
                    if i == len(DataList[:, 1]) - 2:
                        table.append(int(DataList[i + 1, 0]))
                        tables.append(table)
                except:
                    table.append(DataList[i, 0])
                    if DataList[i, 1] != DataList[i + 1, 1]:
                        tables.append(table)
                        table = []
                    if i == len(DataList[:, 1]) - 2:
                        table.append(DataList[i + 1, 0])
                        tables.append(table)
            subfiles_classify = []
            for i in tables:
                subfilename = []
                for k in subfiles:
                    if k['ifile'] in i:
                        subfilename.append(k)
                subfiles_classify.append(subfilename)

            return subfiles_classify

    @staticmethod
    def csp_crc32_memory(data):
        """
        Calculate CRC32 checksum for data validation
        
        Parameters
        ----------
        data : bytes
            Binary data to calculate checksum for

        Returns
        -------
        int
            CRC32 checksum value
        """
        crc_tab = [
            0x00000000, 0xF26B8303, 0xE13B70F7, 0x1350F3F4, 0xC79A971F, 0x35F1141C, 0x26A1E7E8, 0xD4CA64EB,
            0x8AD958CF, 0x78B2DBCC, 0x6BE22838, 0x9989AB3B, 0x4D43CFD0, 0xBF284CD3, 0xAC78BF27, 0x5E133C24,
            0x105EC76F, 0xE235446C, 0xF165B798, 0x030E349B, 0xD7C45070, 0x25AFD373, 0x36FF2087, 0xC494A384,
            0x9A879FA0, 0x68EC1CA3, 0x7BBCEF57, 0x89D76C54, 0x5D1D08BF, 0xAF768BBC, 0xBC267848, 0x4E4DFB4B,
            0x20BD8EDE, 0xD2D60DDD, 0xC186FE29, 0x33ED7D2A, 0xE72719C1, 0x154C9AC2, 0x061C6936, 0xF477EA35,
            0xAA64D611, 0x580F5512, 0x4B5FA6E6, 0xB93425E5, 0x6DFE410E, 0x9F95C20D, 0x8CC531F9, 0x7EAEB2FA,
            0x30E349B1, 0xC288CAB2, 0xD1D83946, 0x23B3BA45, 0xF779DEAE, 0x05125DAD, 0x1642AE59, 0xE4292D5A,
            0xBA3A117E, 0x4851927D, 0x5B016189, 0xA96AE28A, 0x7DA08661, 0x8FCB0562, 0x9C9BF696, 0x6EF07595,
            0x417B1DBC, 0xB3109EBF, 0xA0406D4B, 0x522BEE48, 0x86E18AA3, 0x748A09A0, 0x67DAFA54, 0x95B17957,
            0xCBA24573, 0x39C9C670, 0x2A993584, 0xD8F2B687, 0x0C38D26C, 0xFE53516F, 0xED03A29B, 0x1F682198,
            0x5125DAD3, 0xA34E59D0, 0xB01EAA24, 0x42752927, 0x96BF4DCC, 0x64D4CECF, 0x77843D3B, 0x85EFBE38,
            0xDBFC821C, 0x2997011F, 0x3AC7E3EB, 0xC8AC71E8, 0x1C661503, 0xEE0D9600, 0xFD5D65F4, 0x0F36E6F7,
            0x61C69362, 0x93AD1061, 0x80FDE395, 0x72966096, 0xA65C047D, 0x5437877E, 0x4767748A, 0xB50CF789,
            0xEB1FCBAD, 0x197448AE, 0x0A24BB5A, 0xF84F3859, 0x2C855CB2, 0xDEEEDFB1, 0xCDBE2C45, 0x3FD5AF46,
            0x7198540D, 0x83F3D70E, 0x90A324FA, 0x62C8A7F9, 0xB602C312, 0x44694011, 0x5739B3E5, 0xA55230E6,
            0xFB410CC2, 0x092A8FC1, 0x1A7A7C35, 0xE811FF36, 0x3CDB9BDD, 0xCEB018DE, 0xDDE0EB2A, 0x2F8B6829,
            0x82F63B78, 0x709DB87B, 0x63CD4B8F, 0x91A6C88C, 0x456CAC67, 0xB7072F64, 0xA457DC90, 0x563C5F93,
            0x082F63B7, 0xFA44E0B4, 0xE9141340, 0x1B7F9043, 0xCFB5F4A8, 0x3DDE77AB, 0x2E8E845F, 0xDCE5075C,
            0x92A8FC17, 0x60C37F14, 0x73938CE0, 0x81F80FE3, 0x55326B08, 0xA759E80B, 0xB4091BFF, 0x466298FC,
            0x1871A4D8, 0xEA1A27DB, 0xF94AD42F, 0x0B21572C, 0xDFEB33C7, 0x2D80B0C4, 0x3ED04330, 0xCCBBC033,
            0xA24BB5A6, 0x502036A5, 0x4370C551, 0xB11B4652, 0x65D122B9, 0x97BAA1BA, 0x84EA524E, 0x7681D14D,
            0x2892ED69, 0xDAF96E6A, 0xC9A99D9E, 0x3BC21E9D, 0xEF087A76, 0x1D63F975, 0x0E330A81, 0xFC588982,
            0xB21572C9, 0x407EF1CA, 0x532E023E, 0xA145813D, 0x758FE5D6, 0x87E466D5, 0x94B49521, 0x66DF1622,
            0x38CC2A06, 0xCAA7A905, 0xD9F75AF1, 0x2B9CD9F2, 0xFF56BD19, 0x0D3D3E1A, 0x1E6DCDEE, 0xEC064EED,
            0xC38D26C4, 0x31E6A5C7, 0x22B65633, 0xD0DDD530, 0x0417B1DB, 0xF67C32D8, 0xE52CC12C, 0x1747422F,
            0x49547E0B, 0xBB3FFD08, 0xA86F0EFC, 0x5A048DFF, 0x8ECEE914, 0x7CA56A17, 0x6FF599E3, 0x9D9E1AE0,
            0xD3D3E1AB, 0x21B862A8, 0x32E8915C, 0xC083125F, 0x144976B4, 0xE622F5B7, 0xF5720643, 0x07198540,
            0x590AB964, 0xAB613A67, 0xB831C993, 0x4A5A4A90, 0x9E902E7B, 0x6CFBAD78, 0x7FAB5E8C, 0x8DC0DD8F,
            0xE330A81A, 0x115B2B19, 0x020BD8ED, 0xF0605BEE, 0x24AA3F05, 0xD6C1BC06, 0xC5914FF2, 0x37FACCF1,
            0x69E9F0D5, 0x9B8273D6, 0x88D28022, 0x7AB90321, 0xAE7367CA, 0x5C18E4C9, 0x4F48173D, 0xBD23943E,
            0xF36E6F75, 0x0105EC76, 0x12551F82, 0xE03E9C81, 0x34F4F86A, 0xC69F7B69, 0xD5CF889D, 0x27A40B9E,
            0x79B737BA, 0x8BDCB4B9, 0x988C474D, 0x6AE7C44E, 0xBE2DA0A5, 0x4C4623A6, 0x5F16D052, 0xAD7D5351
        ]
        crc = 0xFFFFFFFF
        for i in range(len(data)):
            crc = crc_tab[(crc ^ data[i]) & 0xff] ^ (crc >> 8)
        return crc ^ 0xFFFFFFFF

    @staticmethod
    def intertelem(telem, datas):
        """
        Interpolate telemetry data for each event
        
        Parameters
        ----------
        telem : list
            Telemetry data for each channel
        datas : list
            Event data for each channel

        Returns
        -------
        datas : list
            Event data with interpolated telemetry values
        """
        for i in range(4):
            Min = (datas[i][:, 0].min() - 1)
            Max = (datas[i][:, 0].max() + 1)
            P = (telem[i][:, 0] < Max) == (telem[i][:, 0] > Min)
            datas_telem = telem[i][P]
            if not len(datas_telem):
                st.warning('Telemetering data interpolation failure')
                return 0
            # print(datas_telem)
            V_tck = interpolate.splrep(datas_telem[:, 0], datas_telem[:, 1], k=1)
            T_tck = interpolate.splrep(datas_telem[:, 0], datas_telem[:, 2], k=1)
            V = interpolate.splev(datas[i][:, 0], V_tck, der=0)
            T = interpolate.splev(datas[i][:, 0], T_tck, der=0)
            datas[i] = np.hstack([datas[i], V.reshape((-1, 1)), T.reshape((-1, 1))])

        return datas

    def extract(self):
        """
        Main data extraction function dispatcher
        
        Calls the appropriate extraction method based on detector type
        
        Returns
        -------
        list
            Paths to extracted data files
        """
        if self.detector == '天宁01':
            return self.extract01()
        elif self.detector == '天宁02':
            return self.extract02()

    def extract01(self):
        """
        Data extraction for TianNing-01 detector
        
        Processes all data files for TianNing-01, extracts science data,
        applies time calibration, and saves results.
        
        Returns
        -------
        extract_files : list
            Paths to the extracted data files
        """
        extract_files = []

        for dat_file in self.datfile:
            print(dat_file)
            st.subheader('正在处理子文件:\n' + str(dat_file).split('Source')[1])
            subfiles = fin.find_subfile(dat_file)
            self.logger.info(f'Parser:extract01:subfiles:{subfiles}')
            '''
            [[{'ifile': 4, 'filename': '.\\DownloadData_TianNing-01\\Source\\\\TN01-20221129-测试结果\\\\1\\file-4_SCI.bin',
               'type': 'SCI'}
              {'ifile': 5,
               'filename': '.\\DownloadData_TianNing-01\\Source\\\\TN01-20221129-测试结果\\\\1\\file-5_TIME.bin',
               'type': 'TIME'}]
             [{'ifile': 7, 'filename': '.\\DownloadData_TianNing-01\\Source\\\\TN01-20221129-测试结果\\\\1\\file-7_SCI.bin',
               'type': 'SCI'}
             {'ifile': 8, 'filename': '.\\DownloadData_TianNing-01\\Source\\\\TN01-20221129-测试结果\\\\1\\file-8_TIME.bin',
              'type': 'TIME'}]]
            '''

            with st.spinner('正在进行数据提取...'):
                telems = [np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3))]
                if not self.csvfile:
                    for i in subfiles:
                        if i['type'] == 'HK':
                            telem = self.parser_HK(i)

                            for k in range(4):
                                telems[k] = np.vstack([telems[k], telem[k]])
                else:
                    for telemfile in self.csvfile:
                        telem = self.parser_Telem(telemfile)
                        for i in range(4):
                            telems[i] = np.vstack([telems[i], telem[i]])
                            # telems = np.delete(telems,[0],axis=0)
                    self.logger.info(f'Parser:extract01:telems:{telems}')
                subfiles_classify = self.parser_LOG(subfiles, '排序')
                self.logger.info(f'Parser:extract01:subfiles_classify:{subfiles_classify}')
                if True:
                    # if subfiles_classify != []:
                    subfiles_classify = np.array(subfiles_classify)

                    n = 0
                    for subfile_classify in subfiles_classify:
                        for i in subfile_classify:
                            if i['type'] == 'TIME':
                                h, k, b = self.parser_time(i)
                            if i['type'] == 'SCI':
                                datas = self.parser_sci(i)
                        for i in range(4):
                            datas[i][:, 0] = k * datas[i][:, 0] + b

                        datas = self.intertelem(telems, datas)

                        output_path = dat_file.parents[2] / 'Output' / dat_file.parent.name / dat_file.stem
                        extract_path = dat_file.parents[2] / 'Output' / dat_file.parent.name / dat_file.stem / 'Extract'
                        fix_path = dat_file.parents[2] / 'Output' / dat_file.parent.name / dat_file.stem / 'Fix'
                        output_path.mkdir(exist_ok=True)
                        extract_path.mkdir(exist_ok=True)
                        fix_path.mkdir(exist_ok=True)
                                               
                        
                        try:
                            hdu = [save.HDUMaker(HDUtype='PrimaryHDU')]
                            for i in range(4):
                                hdu.append(save.HDUMaker(f'CH{i}_EXTRACT_DATAS',
                                                            [('Creator', 'SCUGRID', 'Sichuan University GRID team'),
                                                            (
                                                                'FileTime',
                                                                datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                                                                'file created time')],
                                                            [('UTC', datas[i][:, 0], 'K', 's'),
                                                            ('ADC', datas[i][:, 1], 'E', 'ADC'),
                                                            ('Bias', datas[i][:, 2], 'E', 'V'),
                                                            ('Tem', datas[i][:, 3], 'E', 'Degree')]))
                            fits_path = dat_file.parents[2] / r'Output' / dat_file.relative_to(dat_file.parents[1]).with_suffix('') / 'Extract' / f'Observation_mission-{n}.fits'
                            save.FitsSaver(fits_path, hdu)
                            # np.save(dat_file.split('.dat')[0]+r'_Analyse\\'+f'Extract_Observation_mission-{n}', datas)
                            extract_files.append(fits_path)
                        except:
                            continue

                        n += 1
                st.success('数据提取完成')

        return extract_files

    def extract02(self):
        """
        Data extraction for TianNing-02 detector
        
        Processes all data files for TianNing-02, extracts science data,
        applies time calibration, and saves results.
        
        Returns
        -------
        extract_files : list
            Paths to the extracted data files
        """
        extract_files = []

        cl, Type = fin.cl02(self.datfile)
        for m in range(len(cl)):
            subfiles = fin.find_subfile02(cl[m])

            with st.spinner('正在进行数据提取...'):
                telems = [np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3))]
                for i in subfiles:
                    if i['type'] == 'HK':
                        telem = self.parser_HK(i)
                        for i in range(4):
                            telems[i] = np.vstack([telems[i], telem[i]])

                # print(telems)
                subfiles_classify = self.parser_LOG(subfiles, '排序')

                # if subfiles_classify:
                subfiles_classify = np.array(subfiles_classify)
                # st.write(subfiles_classify)

                n = 0
                for subfile_classify in subfiles_classify:
                    for i in subfile_classify:
                        if i['type'] == 'TIME':
                            h, k, b = self.parser_time(i)
                        if i['type'] == 'SCI':
                            datas = self.parser_sci(i)
                    for i in range(4):
                        datas[i][:, 0] = k * datas[i][:, 0] + b

                    datas = self.intertelem(telems, datas)

                    # print(self.filepath)
                    extract_path = self.filepath.parents[1] / 'Output' / self.filepath.name / Type[m] / 'Extract'
                    fix_path = self.filepath.parents[1] / 'Output' / self.filepath.name / Type[m] / 'Fix'
                    science_path = self.filepath.parents[1] / 'Output' / self.filepath.name / Type[m] / 'Science'
                    extract_path.mkdir(exist_ok=True, parents=True)
                    fix_path.mkdir(exist_ok=True, parents=True)
                    science_path.mkdir(exist_ok=True, parents=True)

                    if datas != 0:
                        hdu = [save.HDUMaker(HDUtype='PrimaryHDU')]
                        for i in range(4):
                            hdu.append(save.HDUMaker(f'CH{i}_EXTRACT_DATAS',
                                                        [('Creator', 'SCUGRID', 'Sichuan University GRID team'),
                                                        (
                                                            'FileTime',
                                                            datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                                                            'file created time')],
                                                        [('UTC', datas[i][:, 0], 'D', 's'),
                                                        ('ADC', datas[i][:, 1], 'E', 'ADC'),
                                                        ('Bias', datas[i][:, 2], 'E', 'V'),
                                                        ('Tem', datas[i][:, 3], 'E', 'Degree')]))
                        fits_path = extract_path / f'Observation_mission-{n}.fits'
                        save.FitsSaver(fits_path, hdu)
                        extract_files.append(fits_path)
                    n += 1
            st.success('数据提取完成')

        return extract_files

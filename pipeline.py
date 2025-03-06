import os
import re
import struct
from pathlib import Path

import numpy as np
import plotly.io as pio
import streamlit as st
from matplotlib import pyplot as plt

import fix as fix
import plots as t
import save as save
import search as fin
from parse import Parser

class Pipline:
    """
    Main pipeline class to orchestrate data processing for GRID satellite data
    
    This class handles data extraction, processing, and visualization for the
    TianNing satellite gamma ray detector data.
    """
    
    def __init__(self, filepaths, detector, logger):
        """
        Initialize the Pipeline object
        
        Parameters
        ----------
        filepaths : dict
            Dictionary mapping detector names to file paths
        detector : str
            Detector name ('天宁01' or '天宁02')
        logger : Logger
            Logger object for recording processing information
        """
        self.detector = detector
        self.filepaths = filepaths
        self.filepath = Path(self.filepaths[detector])
        self.logger = logger

    def Process(self):
        """
        Process new observation data
        
        Identifies unprocessed observation folders, extracts data, applies fixes,
        and saves the results. Updates the log file when processing is complete.
        
        Returns
        -------
        int
            Status code (0 for success)
        """
        log_path = self.filepath / 'log.txt'
        file = []
        unproduced_folders = []
        log = []

        # Read the log file to determine which folders have already been processed
        with open(log_path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                log.append(line)
        
        # Find folders that haven't been processed yet
        for object in self.filepath.iterdir():
            if object.is_dir():
                if object.name not in log:
                    unproduced_folders.append(object.name)
        
        # Display status message if no unprocessed folders were found
        start_run = 0
        if not unproduced_folders:
            st.info("未发现待处理文件，请检查日志文件")
        else:
            # Display list of unprocessed folders and prompt to begin processing
            st.header('发现待处理文件：')
            for unproduced_folder in unproduced_folders:
                st.markdown(f'''
                            -{unproduced_folder}
                            ''')
            start_run = st.button("开始处理")

        # Process each unprocessed folder if start button was pressed
        if start_run:
            for unproduced_folder in unproduced_folders:
                st.header('正在处理:' + unproduced_folder)
                # Create output directory
                output_dir = self.filepath.parent / 'Output' / unproduced_folder
                output_dir.mkdir(exist_ok=True)
                
                with st.spinner("处理中..."):
                    with st.expander('细节'):
                        # Find all data files in the folder
                        dat_files = []
                        csv_files = []
                        catalogue2 = []  # Directory record
                        unproduced_folder_path = self.filepath / unproduced_folder
                        
                        # Collect CSV files
                        for csv_file in unproduced_folder_path.glob('*.csv'):
                            csv_files.append(csv_file)
                            
                        # Collect data files based on detector type
                        if self.detector == '天宁01':
                            for dat_file in unproduced_folder_path.glob('*.dat'):
                                dat_files.append(dat_file)
                        else:
                            for dat_file in unproduced_folder_path.rglob('*.*'):
                                dat_files.append(dat_file)

                        # Check if any data files were found
                        if not len(dat_files):
                            st.warning('Observation data can\'t be found')
                            continue
                            
                        # Log the files found for processing
                        self.logger.info(f'Pipline:Process:datfile：{dat_files}')
                        self.logger.info(f'Pipline:Process:csvfile：{csv_files}')
                        
                        # Process the data: extract, fix, and save
                        parser = Parser(self.logger, csv_files, dat_files,
                                        self.detector, unproduced_folder_path)
                        extract_file = parser.extract()
                        fix_file = fix.fix(extract_file, self.detector)
                        self.logger.info(f'Pipline:Process:FixFile：{fix_file}')
                        save.SaveScience(fix_file, self.detector)

                        # Update the log file with the processed folder
                        with open(log_path, 'a') as f:
                            f.write(unproduced_folder + '\n')

                        # Display visualization of the extraction results
                        st.markdown('## 提取结果：')
                        t.Plot_Extract(extract_file)

                        # Display visualization of the fix results
                        st.markdown('## 能谱修正结果：')
                        t.Plot_Fix(fix_file)

                    # Display completion message
                    st.success(unproduced_folder + "处理完毕")
        return 0

    def Look(self):
        """
        View historical processing results
        
        Allows the user to select a previously processed observation and
        view the extracted and fixed data visualizations.
        
        Returns
        -------
        int
            Status code (0 for success)
        """
        # Paths for log file and figure directories
        log_path = self.filepath.parent / 'Source' / 'log.txt'
        Expath = 'figure/Extract'
        Fixpath = 'figure/Fix'

        log = []

        # Read the log file to get processed folders
        with open(log_path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                log.append(line)

        # If no processed folders were found
        if not log:
            st.info('未找到历史文件')
        else:
            # Allow user to select a processed folder
            selpath = st.selectbox('选择你想要查看的文件', log)
            data_dir = self.filepath / selpath
            
            # Display all subfolders and their visualizations
            for subdir in data_dir.iterdir():
                print(subdir)
                st.markdown(f'# {subdir.name}')
                with st.spinner('载入中...'):
                    # Display extraction results
                    st.markdown('### 提取结果：')
                    expath_full = subdir / Expath
                    print('paths:', expath_full)
                    for json in expath_full.glob('*.json'):
                        fig = pio.read_json(json)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display fix results
                    fix_path_full = subdir / Fixpath
                    st.markdown('### 修正结果：')
                    for json in fix_path_full.glob('*.json'):
                        fig = pio.read_json(json)
                        st.plotly_chart(fig, use_container_width=True)
        return 0

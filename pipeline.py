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
    def __init__(self, filepaths, detector, logger):
        self.detector = detector
        self.filepaths = filepaths
        self.filepath = Path(self.filepaths[detector])
        self.logger = logger

    def Process(self):
        log_path = self.filepath / 'log.txt'
        file = []
        unproduced_folders = []
        log = []

        with open(log_path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                log.append(line)
        
        for object in self.filepath.iterdir():
            if object.is_dir():
                if object.name not in log:
                    unproduced_folders.append(object.name)
            # file.append(i)
        # self.logger.info(f'file[0]：{file[0]}')
        # self.logger.info(f'Log：{log}')
        # 待处理文件添加
        # self.logger.info(f'unprofile：{unproduced_folders}')
        start_run = 0

        if not unproduced_folders:
            st.info("未发现待处理文件，请检查日志文件")
        else:
            st.header('发现待处理文件：')
            for unproduced_folder in unproduced_folders:
                st.markdown(f'''
                            -{unproduced_folder}
                            ''')
            start_run = st.button("开始处理")

        if start_run:
            for unproduced_folder in unproduced_folders:
                st.header('正在处理:' + unproduced_folder)
                # 创建同名文件夹
                output_dir = self.filepath.parent / 'Output' / unproduced_folder
                    # if not os.path.exists(self.filepath.parent / 'Output' / i):
                    #     os.mkdir(self.filepath.parent + 'Output' + i)
                output_dir.mkdir(exist_ok=True)
                with st.spinner("处理中..."):
                    with st.expander('细节'):
                        dat_files = []
                        csv_files = []
                        catalogue2 = []  # 目录，记载
                        unproduced_folder_path = self.filepath / unproduced_folder
                        
                        for csv_file in unproduced_folder_path.glob('*.csv'):
                            csv_files.append(csv_file)
                        if self.detector == '天宁01':
                            for dat_file in unproduced_folder_path.glob('*.dat'):
                                dat_files.append(dat_file)
                        else:
                            for dat_file in unproduced_folder_path.rglob('*.*'):
                                dat_files.append(dat_file)
                        
                        # for k in os.walk(self.filepath / unproduced_folder):
                        #     catalogue2.append(k)

                        # for k in catalogue2[0][2]:
                        #     if os.path.splitext(k)[-1] == '.csv':
                        #         csv_file.append(catalogue2[0][0] + r'\\' + k)
                        #     else:
                        #         dat_file.append(catalogue2[0][0] + r'\\' + k)

                        if not len(dat_files):
                            st.warning('Observation data can\'t be found')
                            continue
                        self.logger.info(f'Pipline:Process:datfile：{dat_files}')
                        self.logger.info(f'Pipline:Process:csvfile：{csv_files}')
                        parser = Parser(self.logger, csv_files, dat_files,
                                        self.detector, unproduced_folder_path)
                        extract_file = parser.extract()
                        fix_file = fix.fix(extract_file, self.detector)
                        self.logger.info(f'Pipline:Process:FixFile：{fix_file}')
                        save.SaveScience(fix_file, self.detector)

                        with open(log_path, 'a') as f:
                            f.write(unproduced_folder + '\n')

                        st.markdown('## 提取结果：')
                        t.Plot_Extract(extract_file)

                        
                        st.markdown('## 能谱修正结果：')
                        t.Plot_Fix(fix_file)

                    st.success(unproduced_folder + "处理完毕")
        return 0

    def Look(self):
        log_path = self.filepath.parent / 'Source' / 'log.txt'
        Expath = 'figure/Extract'
        Fixpath = 'figure/Fix'

        log = []

        with open(log_path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                log.append(line)

        if not log:
            st.info('未找到历史文件')
        else:
            selpath = st.selectbox('选择你想要查看的文件', log)
            data_dir = self.filepath / selpath
            for subdir in data_dir.iterdir():
                print(subdir)
                st.markdown(f'# {subdir.name}')
                with st.spinner('载入中...'):
                    st.markdown('### 提取结果：')
                    expath_full = subdir / Expath
                    print('paths:', expath_full)
                    for json in expath_full.glob('*.json'):
                        # if img.with_suffix('.json').exists():
                            # Html = img.with_suffix('.json')
                        fig = pio.read_json(json)
                        st.plotly_chart(fig, use_container_width=True)
                        # else:
                        #     Exfig = plt.imread(img)
                        #     st.image(Exfig)
                    fix_path_full = subdir / Fixpath
                    st.markdown('### 修正结果：')
                    for json in fix_path_full.glob('*.json'):
                        # print(img)
                        # if img.with_suffix('.json').exists():
                        #     Html = img.with_suffix('.json')
                        fig = pio.read_json(json)
                        st.plotly_chart(fig, use_container_width=True)
                        # else:
                        #     fix_fig = plt.imread(img)
                        #     st.image(fix_fig)
        return 0

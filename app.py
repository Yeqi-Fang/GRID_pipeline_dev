import datetime
import logging
import zipfile
from pathlib import Path

import streamlit as st

from pipeline import Pipline

# Generate a timestamp for the log file name
now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Configure logging with appropriate format and file location
logging.basicConfig(handlers=[logging.FileHandler(filename=f'.//log//{now}.log', encoding='utf-8')],
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%F %A %T",
                    level=logging.INFO)
logger = logging.getLogger()

# Configure Streamlit settings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

# Set up the main page title and external link
st.title("四川大学卫星下传数据可视化平台")
url = "http://www.scutiange.top/"
st.write("天格卫星轨道可视化界面： [scutiange.top](%s)" % url)

# Create sidebar selection for detector and operation mode
detector = st.sidebar.selectbox("探测器编号", ['天宁01', '天宁02'], key="1")
Work = st.sidebar.selectbox("工作模式", ['卫星下传数据处理', '响应矩阵', '历史处理结果可视化'], key="3")

# Set appropriate paths based on selected operation mode
if Work == '历史处理结果可视化':
    file_path = {'天宁01': Path('./DownloadData_TianNing-01/Output'),
                '天宁02': Path('./DownloadData_TianNing-02/Output')}
else:
    file_path = {'天宁01': Path('./DownloadData_TianNing-01/Source'),
                '天宁02': Path('./DownloadData_TianNing-02/Source')}
                
# Initialize pipeline with the appropriate paths
pipline = Pipline(file_path, detector, logger)

# Create file upload functionality in the sidebar
with st.sidebar:
    uploads = st.file_uploader(
        '上传数据', type=['zip'], accept_multiple_files=True)
    upst = st.button('开始上传')

    # Extract uploaded files when button is clicked
    with st.spinner('上传中...'):
        if uploads is not None and upst:
            for upload in uploads:
                f = zipfile.ZipFile(upload)
                for file in f.namelist():
                    f.extract(file, file_path[detector])

# Execute the appropriate pipeline function based on selected mode
if Work == '卫星下传数据处理':
    pipline.Process()
elif Work == '历史处理结果可视化':
    pipline.Look()


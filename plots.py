"""
Plotting utilities for GRID pipeline data visualization and analysis.
This module provides functions to visualize detector data, including 
light curves, energy spectra, and hardware metrics.
"""

# Standard library imports
import csv
import datetime
import os
import re
import struct

# Data visualization and analysis imports
import matplotlib as mpl
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Custom module imports
import grafica
import save

# Configure matplotlib for Chinese characters
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False

# Define color palette for plots
colors = [(227, 119, 194), (255, 127, 14),
          (44, 160, 44), (214, 39, 40),
          (148, 103, 189), (140, 86, 75),
          (227, 119, 194), (127, 127, 127),
          (188, 189, 34), (23, 190, 207)]


# Define Gaussian function for curve fitting
def f(x, a, b, c):
    """
    Gaussian function for curve fitting.
    
    Parameters:
    - x: Input data point
    - a: Amplitude
    - b: Mean (center)
    - c: Standard deviation
    
    Returns:
    - Gaussian function value at point x
    """
    return a * np.exp(-(x - b) ** 2 / c ** 2)


def Plot_Extract(ExtractFile):
    """
    Generates visualization plots from extracted data files.
    
    Creates multiple visualization tabs with different plots:
    - All plots overview (light curves, spectra, bias voltage, temperature)
    - Light curve (detailed view)
    - Energy spectrum (detailed view)
    - SiPM average bias voltage (detailed view)
    - SiPM average temperature (detailed view)
    
    Parameters:
    - ExtractFile: List of paths to data files to be plotted
    """
    n = 0
    Loadpath = 'figure/Extract'

    for filepath in ExtractFile:
        # Read data from the file
        datas = save.ReadMid(filepath)

        # Create tabs for different visualizations
        mat, tab1, tab2, tab3, tab4 = st.tabs(
            ["全部图形", "光变曲线", "能谱图", "SiPM平均偏压", "SiPM平均温度"])
        with mat:
            # Overview of all plots
            total = np.array([0])
            for plotter in grafica.manager.plotters:
                fig = grafica.manager.new(
                    plotter_name=plotter,
                    subplots=True
                )
                for k in range(4):
                    fig.histogram(datas[k][:, 0], bins=np.arange(
                        datas[k][:, 0].min(), datas[k][:, 0].max(), 3), density=False, label=f'通道{k}', row=1, col=1, showlegend=False, color=colors[k])
                    total = np.hstack([total, datas[k][:, 0]])
                fig.histogram(np.delete(total, [0], axis=0), bins=np.arange(
                    datas[k][:, 0].min(), datas[k][:, 0].max(), 3), density=False, label='总计', row=1, col=1, showlegend=False, color=colors[4])
                total = np.array([0])
                index = 1
                for k in range(4):
                    fig.histogram(datas[k][:, index], bins=np.arange(
                        datas[k][:, index].min(), datas[k][:, index].max(), 3), density=False, label=f'通道{k}', row=1, col=2, color=colors[k])
                    total = np.hstack([total, datas[k][:, index]])
                fig.histogram(np.delete(total, [0], axis=0), bins=np.arange(
                    datas[k][:, index].min(), datas[k][:, index].max(), 3), density=False, label='总计', row=1, col=2, color=colors[4])

                index = 2
                plotly_fig = fig.plotly_figure
                for k in range(4):
                    plotly_fig.add_trace(go.Scatter(x=datas[k][:, 0],
                                                    y=datas[k][:, index], name=None, showlegend=False), col=1, row=2)

                index = 3
                plotly_fig = fig.plotly_figure
                for k in range(4):
                    plotly_fig.add_trace(go.Scatter(x=datas[k][:, 0], y=datas[k][:, index], showlegend=False), col=2, row=2)
            plotly_fig.update_xaxes(title='UTC时间/秒', row=1, col=1)
            plotly_fig.update_yaxes(title='计数', row=1, col=1)
            plotly_fig.update_xaxes(title='能量/keV', row=1, col=2, type='log')
            plotly_fig.update_yaxes(title='计数', row=1, col=2)
            plotly_fig.update_xaxes(title='UTC时间/秒', row=2, col=1)
            plotly_fig.update_yaxes(title='SiPM平均偏压/伏特', row=2, col=1)
            plotly_fig.update_xaxes(title='UTC时间/秒', row=2, col=2)
            plotly_fig.update_yaxes(title='SiPM平均温度/摄氏度', row=2, col=2)
            plotly_fig.update_layout(height=600)
            file = filepath.parents[1]

            extract_path = file / 'figure' / 'Extract'
            extract_path.mkdir(exist_ok=True, parents=True)
            
            n += 1
            plotly_fig.write_json(extract_path / f'{n}.json')
            st.plotly_chart(plotly_fig, use_container_width=True)

        # Light curve visualization
        with tab1:
            total = np.array([0])
            for plotter in grafica.manager.plotters:
                fig = grafica.manager.new(
                    plotter_name=plotter
                )
                for k in range(4):
                    fig.histogram(datas[k][:, 0], bins=np.arange(
                        datas[k][:, 0].min(), datas[k][:, 0].max(), 3), density=False, label=f'通道{k}')
                    total = np.hstack([total, datas[k][:, 0]])
                fig.histogram(np.delete(total, [0], axis=0), bins=np.arange(
                    datas[k][:, 0].min(), datas[k][:, 0].max(), 3), density=False, label='总计')

            plotly_fig = fig.plotly_figure

            plotly_fig.update_layout(xaxis_title='UTC时间/秒', yaxis_title='计数')
            st.plotly_chart(plotly_fig, use_container_width=True)

        # Energy spectrum visualization
        with tab2:
            total = np.array([0])
            for plotter in grafica.manager.plotters:
                fig = grafica.manager.new(
                    plotter_name=plotter,
                )
                for k in range(4):
                    fig.histogram(datas[k][:, 1], bins=np.arange(
                        datas[k][:, 1].min(), datas[k][:, 1].max(), 3), density=False, label=f'通道{k}')
                    total = np.hstack([total, datas[k][:, 1]])
                fig.histogram(np.delete(total, [0], axis=0), bins=np.arange(
                    datas[k][:, 1].min(), datas[k][:, 1].max(), 3), density=False, label='总计')

            plotly_fig = fig.plotly_figure
            plotly_fig.update_xaxes(type='log')
            plotly_fig.update_layout(xaxis_title='波形峰值', yaxis_title='计数')
            st.plotly_chart(plotly_fig, use_container_width=True)

        # SiPM bias voltage visualization
        with tab3:
            index = 2
            fig = go.Figure()
            for k in range(4):
                fig.add_scatter(x=datas[k][:, 0],
                                y=datas[k][:, index], name=f'通道{k}')
            fig.add_hline(y=28.5, line_dash="dot", line_color="black",
                          annotation_text="28.5V", annotation_position="bottom right")
            fig.update_yaxes(range=[28.4, 28.6])
            fig.update_layout(xaxis_title='UTC时间/秒',
                              yaxis_title='SiPM平均偏压/伏特', title='SiPM偏压')
            st.plotly_chart(fig, use_container_width=True)

        # SiPM temperature visualization  
        with tab4:
            index = 3
            fig = go.Figure()
            for k in range(4):
                fig.add_scatter(x=datas[k][:, 0],
                                y=datas[k][:, index], name=f'通道{k}')
            fig.update_layout(xaxis_title='UTC时间/秒',
                              yaxis_title='SiPM平均温度/摄氏度', title='SiPM平均温度')
            st.plotly_chart(fig, use_container_width=True)


def Plot_Fix(FixFile):
    """
    Generates calibration plots from calibration data files.
    
    Creates visualizations with both Matplotlib and Plotly:
    - Spectral peak analysis
    - Energy calibration metrics
    - Peak position and relative error calculation
    
    Parameters:
    - FixFile: List of paths to calibration data files
    """
    n = 0
    
    for filepath in FixFile:
        # Read calibration data
        Array = save.ReadMid(filepath)
        tab1, tab2 = st.tabs(["Plotly", "Matplotlib"])

        # Matplotlib visualization with peak fitting
        with tab2:
            plt.figure(figsize=(12, 6), dpi=600)
            plt.title(
                f'观测时间: {datetime.datetime.fromtimestamp(int(Array[0][:, 0].min()) + 28800)}')
            plt.xscale(r'log')
            plt.xlabel(r'能量/keV')
            plt.ylabel(r'计数')
            plt.xlim([20, 800])
            h0 = plt.hist(Array[0][:, 1], bins=np.arange(
                0, 1900, 0.5), histtype='step')
            for i in range(4):
                h = plt.hist(Array[i][:, 1], bins=np.arange(
                    0, 1900, 0.5), histtype='step', label=f'通道{i}')
                x = h[1][30:208]
                y = h[0][30:208]
                ymax_index = np.argmax(y)
                c = np.array([y[ymax_index], x[ymax_index], 2])
                popt, pcov = curve_fit(f, x, y, c, maxfev=10000)

                plt.text(125, h0[0].max() - (i + 0.5) * h0[0].max() / 5,
                         f'通道{i}峰值能量: {(popt[1] + 0.25):.3f}' + r'$keV$' + '\n' + f'相对误差: {abs(1 - (popt[1] + 0.25) / 59.5) * 100:.3f}%')
                plt.legend()
            
            fix_path = filepath.parents[1] / 'figure' / 'Fix'
            fix_path.mkdir(exist_ok=True, parents=True)

            plt.savefig(fix_path / f'{n}.png', dpi=200)
            plt.savefig(fix_path / f'{n}.pdf')

            st.pyplot()

        # Interactive Plotly visualization with data table
        with tab1:
            # Create dataframe to store peak analysis results
            df = pd.DataFrame(columns=['峰值能量', '相对误差'])
            for plotter in grafica.manager.plotters:
                fig = grafica.manager.new(
                    plotter_name=plotter,
                )
                for i in range(4):
                    fig.histogram(Array[i][:, 1], bins=np.arange(
                        0, 1900, 0.5), density=False, label=f'通道{i}')
                    h = plt.hist(Array[i][:, 1], bins=np.arange(
                        0, 1900, 0.5), histtype='step', label=f'通道{i}')
                    x = h[1][30:208]
                    y = h[0][30:208]
                    ymax_index = np.argmax(y)
                    c = np.array([y[ymax_index], x[ymax_index], 2])
                    popt, pcov = curve_fit(f, x, y, c, maxfev=10000)
                    df = pd.concat([df, pd.DataFrame({
                        '峰值能量': [f'{(popt[1] + 0.25):.3f} Kev'],
                        '相对误差': [f'{abs(1 - (popt[1] + 0.25) / 59.5) * 100:.3f}%']
                    })], axis=0, ignore_index=True)
            plotly_fig = fig.plotly_figure
            plotly_fig.update_xaxes(
                type='log', title=r'能量/keV', range=[np.log10(20), np.log10(800)])
            plotly_fig.update_yaxes(title='计数')
            st.plotly_chart(plotly_fig, use_container_width=True)
            plotly_fig.write_json(fix_path / f'{n}.json')
            df = df.reset_index(names=['通道'])
            st.write(df)

        n += 1

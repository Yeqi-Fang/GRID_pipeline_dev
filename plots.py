import csv
import datetime
import os
import re
import struct

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

import grafica
import save

mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False


colors = [(227, 119, 194), (255, 127, 14),
          (44, 160, 44), (214, 39, 40),
          (148, 103, 189), (140, 86, 75),
          (227, 119, 194), (127, 127, 127),
          (188, 189, 34), (23, 190, 207)]


# plt.rc('font', family='Times New Roman')


def f(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / c ** 2)

# @st.cache


def Plot_Extract(ExtractFile):
    n = 0
    Loadpath = 'figure/Extract'

    for filepath in ExtractFile:

        datas = save.ReadMid(filepath)

        mat, tab1, tab2, tab3, tab4 = st.tabs(
            ["全部图形", "光变曲线", "能谱图", "SiPM平均偏压", "SiPM平均温度"])
        with mat:
            # plt.figure(figsize=(16, 8))
            # for i in range(4):
            #     plt.subplot(221 + i)
            #     if i < 2:
            #         if i == 1:
            #             plt.xscale(r'log')
            #             plt.title(r"能谱图")
            #             plt.xlabel(r'波形峰值')
            #             plt.ylabel(r'计数')
            #         if i == 0:
            #             # plt.ylim(0,1600)
            #             plt.title("光变曲线")
            #             plt.xlabel('UTC时间/秒')
            #             plt.ylabel('计数')
            #         total = np.array([0])
            #         for k in range(4):
            #             plt.hist(datas[k][:, i], bins=np.arange(datas[k][:, i].min(), datas[k][:, i].max(), 3),
            #                      histtype='step', label=f'通道{k}')
            #             total = np.hstack([total, datas[k][:, i]])
            #         plt.hist(np.delete(total, [0], axis=0), bins=np.arange(datas[k][:, i].min(), datas[k][:, i].max(), 3),
            #                  histtype='step', label='总计')
            #         plt.legend()
            #     else:
            #         for k in range(4):
            #             plt.plot(datas[k][:, 0], datas[k][:, i])
            #         if i == 2:
            #             plt.plot(datas[0][:, 0], 28.5 *
            #                      np.ones(np.size(datas[0][:, 0])), 'k--')
            #             plt.xlabel(r'UTC时间/秒')
            #             plt.ylabel(r'SiPM平均偏压/伏特')
            #             plt.ylim([28.4, 28.6])
            #         if i == 3:
            #             plt.xlabel(r'UTC时间/秒')
            #             plt.ylabel(r'SiPM平均温度/摄氏度')

            # file = filepath.split('Extract')[0]

            # if not os.path.exists(file + r'\figure'):
            #     os.mkdir(file + r'\figure')
            # if not os.path.exists(file + Loadpath):
            #     os.mkdir(file + Loadpath)

            # plt.savefig(file + Loadpath + r'\\' + f'{n}.png', dpi=200)
            # plt.savefig(file + Loadpath + r'\\' + f'{n}.pdf')
            # n += 1

            # st.pyplot()
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
                # fig.plotly_figure.update_traces(line={'width': 1.5})
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
                # plotly_fig.add_hline(y=28.5, line_dash="dot", line_color="black",
                #             annotation_text="28.5V", annotation_position="bottom right")
                # plotly_fig.update_yaxes(range=[28.4, 28.6])
                # plotly_fig.update_layout(xaxis_title='UTC时间/秒',
                #               yaxis_title='SiPM平均偏压/伏特', title='SiPM偏压')

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
            # plotly_fig.update_traces(line={'width': 1.5})
            plotly_fig.update_layout(height=600)
            file = filepath.parents[1]

            # if not os.path.exists(file + r'\figure'):
            #     os.mkdir(file + r'\figure')
            # if not os.path.exists(file + Loadpath):
            #     os.mkdir(file + Loadpath)

            extract_path = file / 'figure' / 'Extract'
            extract_path.mkdir(exist_ok=True, parents=True)
            
            # plt.savefig(file + Loadpath + r'\\' + f'{n}.png', dpi=200)
            # plt.savefig(file + Loadpath + r'\\' + f'{n}.pdf')
            n += 1
            # plotly_fig.update_layout(xaxis_title='UTC时间/秒', yaxis_title='计数')
            plotly_fig.write_json(extract_path / f'{n}.json')
            st.plotly_chart(plotly_fig, use_container_width=True)

        # 光变曲线
        with tab1:
            # total = np.array([0])
            # for k in range(4):
            #     plt.hist(datas[k][:, 0], bins=np.arange(datas[k][:, 0].min(), datas[k][:, 0].max(), 3),
            #              histtype='step', label=f'通道{k}')
            #     total = np.hstack([total, datas[k][:, 0]])
            # plt.hist(np.delete(total, [0], axis=0), bins=np.arange(datas[k][:, 0].min(), datas[k][:, 0].max(), 3),
            #          histtype='step', label='总计')
            # plt.legend()
            # plt.xscale('log')
            # st.pyplot()

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

        # 能谱图
        with tab2:
            # total = np.array([0])
            # for k in range(4):
            #     plt.hist(datas[k][:, 1], bins=np.arange(datas[k][:, 1].min(), datas[k][:, 1].max(), 3),
            #              histtype='step', label=f'通道{k}')
            #     total = np.hstack([total, datas[k][:, 0]])
            # plt.hist(np.delete(total, [0], axis=0), bins=np.arange(datas[k][:, 1].min(), datas[k][:, 1].max(), 3),
            #          histtype='step', label='总计')
            # st.pyplot()

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

        with tab3:
            # total = np.array([0])

            index = 2
            # for k in range(4):
            #     plt.plot(datas[k][:, 0], datas[k][:, index])
            #     plt.plot(datas[0][:, 0], 28.5 *
            #              np.ones(np.size(datas[0][:, 0])), 'k--')
            #     plt.xlabel(r'UTC时间/秒')
            #     plt.ylabel(r'SiPM平均偏压/伏特')
            #     plt.ylim([28.4, 28.6])
            # st.pyplot()

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

        with tab4:
            # total = np.array([0])
            index = 3
            # for k in range(4):
            #     plt.plot(datas[k][:, 0], datas[k][:, index])
            #     plt.xlabel(r'UTC时间/秒')
            #     plt.ylabel(r'SiPM平均温度/摄氏度')
            # st.pyplot()
            fig = go.Figure()
            for k in range(4):
                fig.add_scatter(x=datas[k][:, 0],
                                y=datas[k][:, index], name=f'通道{k}')
            fig.update_layout(xaxis_title='UTC时间/秒',
                              yaxis_title='SiPM平均温度/摄氏度', title='SiPM平均温度')
            st.plotly_chart(fig, use_container_width=True)

# @st.cache


def Plot_Fix(FixFile):
    n = 0
    # Loadpath = r'\figure\Fix'
    for filepath in FixFile:
        Array = save.ReadMid(filepath)
        tab1, tab2 = st.tabs(["Plotly", "Matplotlib"])

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

            # file = filepath.split('Fix')[0]
            # if not os.path.exists(file + r'\figure'):
            #     os.mkdir(file + r'\figure')
            # if not os.path.exists(file + Loadpath):
            #     os.mkdir(file + Loadpath)
            
            fix_path = filepath.parents[1] / 'figure' / 'Fix'
            fix_path.mkdir(exist_ok=True, parents=True)

            plt.savefig(fix_path / f'{n}.png', dpi=200)
            plt.savefig(fix_path / f'{n}.pdf')

            st.pyplot()

        with tab1:
            df = pd.DataFrame(columns=['峰值能量', '相对误差'])
            for plotter in grafica.manager.plotters:
                fig = grafica.manager.new(
                    plotter_name=plotter,
                )
                for i in range(4):
                    # fig.histogram(samples, label='No args', density=density, bins=10)
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
                # h = plt.hist(Array[i][:, 1], bins=np.arange(0, 1900, 0.5), histtype='step', label=f'通道{i}')
            plotly_fig = fig.plotly_figure
            # plotly_fig.update_layout(title=f'观测时间: {datetime.datetime.fromtimestamp(int(Array[0][:, 0].min()) + 28800)}')
            plotly_fig.update_xaxes(
                type='log', title=r'能量/keV', range=[np.log10(20), np.log10(800)])
            plotly_fig.update_yaxes(title='计数')
            st.plotly_chart(plotly_fig, use_container_width=True)
            plotly_fig.write_json(fix_path / f'{n}.json')
            df = df.reset_index(names=['通道'])
            # df = /
            st.write(df)
            # df

        n += 1

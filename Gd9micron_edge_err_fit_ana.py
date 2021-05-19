import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
import math
import glob
import copy
from scipy.optimize import curve_fit
import random

def chi_sq(obs_vals, exp_vals):
    test_stats = 0
    for obs, exp in zip(obs_vals, exp_vals):
        test_stats += (float(obs) - float(exp)) ** 2 / float(exp)
    return test_stats


def func_error(x, amp, mu, sigma, bias):
    array_differencial = np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
    array_integral = amp / math.sqrt(2*np.pi) * sigma * np.cumsum(array_differencial) + bias
    return array_integral


def create_array_histogram(array, _bins, _range):
    counts, bin_edges = np.histogram(array, bins=_bins, range=_range)
    bin_middles =  0.5 * (bin_edges[1:] + bin_edges[:-1])
    return counts, bin_edges, bin_middles


if __name__ == '__main__':

    # output
    output_dir = "Gd9micron_making_superimposed_graph_ydiv26"
    os.makedirs(f'{output_dir}', exist_ok=True)

    pix_to_micron = 0.05499# 2020.02.28 HENPneutronandemulsionpresentation.pdf

    list_json_file = glob.glob("Gd9micron_making_projection_ydiv26/*.json")
    for f in list_json_file:
        base_name = os.path.basename(f)
        with open(f, 'r') as jsonfile:
            file_info = json.loads(jsonfile.read())
        lam = 9.0 / 0.05499 #file_info['fitting_lambda']
        
        
        phi = file_info['fitting_phi']
        list_peak_x = []
        i_peak = 0
        while True:
            x = lam / (2 * np.pi) * (np.pi * (1 + 4 * i_peak) / 2 - phi)
            if x > len(file_info['array_projection']):
                break
            list_peak_x.append(x)
            i_peak += 1

        array_max = np.argmax(file_info['array_projection'], axis = 0)
        print('arg-max-array_max', array_max)
        left_peak = file_info['array_projection'][0:array_max]
        right_peak = file_info['array_projection'][array_max:-1]
        right_peak.reverse()
        
        print('left_peak',left_peak)
        print('right_peak',right_peak)

        graph_x = np.arange(0, len(file_info['array_projection']))
        graph_x_left = np.arange(0, len(left_peak))
        graph_x_right = np.arange(0, len(right_peak))

        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(graph_x, file_info['array_projection'], marker='o', c = 'blue',linewidth=6)
        ax.axvline(array_max, c = 'm')
        ax.set_title("Y projection")
        ax.set_xlabel("Y")
        ax.set_ylabel("average brightness")
        fig.tight_layout()
        plt.savefig(f"{output_dir}/{base_name.replace('.json', '')}_peaks.png")
        plt.clf()             

        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(graph_x_left, left_peak, marker='o', c = 'blue',linewidth=6)
        ax.set_title("Y projection_left peak")
        ax.set_xlabel("Y")
        ax.set_ylabel("average brightness")
        fig.tight_layout()
        plt.savefig(f"{output_dir}/{base_name.replace('.json', '')}_left_peaks.png")
        plt.clf()             

        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(graph_x_right, right_peak, marker='o', c = 'blue',linewidth=6)
        ax.set_title("Y projection_right peak")
        ax.set_xlabel("Y")
        ax.set_ylabel("average brightness")
        fig.tight_layout()
        plt.savefig(f"{output_dir}/{base_name.replace('.json', '')}_right_peaks.png")
        plt.clf()             


        n_bin = 50
        range_max = len(file_info['array_projection'])
        ndf_fitting = 4

        fitting_initial_vals = [22000, 100, 10, 1000]
        popt, pcov = curve_fit(func_error, graph_x, file_info['array_projection'], p0=fitting_initial_vals)
        #print(popt)
        #print(pcov)
        counts_fit = func_error(graph_x, popt[0], popt[1], popt[2], popt[3])
        this_chisq = chi_sq(file_info['array_projection'], counts_fit)    
        label_str = 'error function: '
        label_str += f'sigma={popt[2]:.2f}+-{math.sqrt(pcov[2][2]):.2f}, '
        label_str += f'Chi^2 = {this_chisq:.2f}, '
        label_str += f'reduced_Chi^2 = {this_chisq / (n_bin - ndf_fitting):.2f}'
        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(1, 1, 1)
        ax.errorbar(graph_x, file_info['array_projection'], np.sqrt(file_info['array_projection']), range_max / n_bin/2, fmt=".", color='g')
        ax.plot(graph_x, counts_fit, '-r', label=label_str)
        ax.set_title("fitting with step-like distribution")
        # for p in list_peak_x:
        #     ax.axvline(p, c = 'r')
        ax.axvline(array_max, c = 'm')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend()
        fig.tight_layout()
        plt.savefig(f"{output_dir}/{base_name.replace('.json', '')}_peaks_fitting.png")
        plt.clf()
        print('list_peak_x',list_peak_x)


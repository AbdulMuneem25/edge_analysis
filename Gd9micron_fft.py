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
    output_dir = "Gd9micron_projection_err_ft_fitting_04"
    os.makedirs(f'{output_dir}', exist_ok=True)

    pix_to_micron = 0.05499# 2020.02.28 HENPneutronandemulsionpresentation.pdf

    list_json_file = glob.glob("Gd9micron_making_projection_foulier/*.json")
    reduce_chisq = []
    sigma_vals = []
    chi_sqvals = []


    for f in list_json_file:
        base_name = os.path.basename(f)
        with open(f, 'r') as jsonfile:
            file_info = json.loads(jsonfile.read())

        list = file_info['phase']
        projection = file_info['array_projection']

        for i in range(len(list)):
            if list[i + 1] > len(projection):
                break
            edge0 = list[i]
            edge1 = list[i+1]
            list_partial = copy.deepcopy(projection[edge0:edge1])
            if i % 2 ==0:
                list_partial.reverse()
                print('revered', i)
            print(list_partial[0], list_partial[-1])
            graph_x = np.arange(0, len(list_partial))
            n_bin = 50
            range_max = len(list_partial)
            ndf_fitting = 4

            fitting_initial_vals = [23000, 30, 20, 5]
            popt, pcov = curve_fit(func_error, graph_x, list_partial, p0=fitting_initial_vals)
            counts_fit = func_error(graph_x, popt[0], popt[1], popt[2], popt[3])
            this_chisq = chi_sq(list_partial, counts_fit)
            reduce_chi =  this_chisq / (n_bin - ndf_fitting)    
            reduce_chisq.append(reduce_chi)
            sigma_vals.append(popt[2])

            label_str = 'error function: '
            label_str += f'sigma={popt[2]*pix_to_micron:.2f}+-{math.sqrt(pcov[2][2]):.2f}, \n'
            label_str += f'Chi^2 = {this_chisq:.2f}, '
            label_str += f'reduced_Chi^2 = {reduce_chi:.2f}'
            fig = plt.figure(figsize=(8.0, 6.0))
            ax = fig.add_subplot(1, 1, 1)
            ax.errorbar(graph_x, list_partial, np.sqrt(list_partial), range_max / n_bin/2, fmt=".", color='m')
            ax.plot(graph_x, counts_fit, '-r', label=label_str)
            ax.set_title("Edge distribution")
            #ax.axvline(min_left_peak, c = 'g')

            ax.set_xlabel("Y")
            ax.set_ylabel("Brightness sum")
            ax.set_xlabel("Pixels")

            ax.legend()
            fig.tight_layout()
            plt.savefig(f"{output_dir}/{base_name.replace('.json', '')}_{i}_fitting.png")
            plt.clf()
            
    sigma_val = []
    for i in sigma_vals:
        sig_pix = i * pix_to_micron
        sigma_val.append(sig_pix)

        
    n_bin = 20
    x_min = 0.0
    x_max = 2

    bin_heights, bin_borders = np.histogram(sigma_val, n_bin, (x_min, x_max))
    bin_middles = 0.5*(bin_borders[1:] + bin_borders[:-1])
    plt.errorbar(bin_middles, bin_heights, np.sqrt(bin_heights), (x_max - x_min) / (n_bin * 2), fmt='o',color='g',label='sigma')
    plt.legend()
    plt.xlabel('Sigma[$\mu$m]')
    plt.ylabel('Counts')

    plt.savefig(f"{output_dir}/sigma_histo.png")
    plt.clf()

    n_bin = 20
    x_min = 0.0
    x_max = 150

    bin_heights, bin_borders = np.histogram(reduce_chisq, n_bin, (x_min, x_max))
    bin_middles = 0.5*(bin_borders[1:] + bin_borders[:-1])
    plt.errorbar(bin_middles, bin_heights, np.sqrt(bin_heights), (x_max - x_min) / (n_bin * 2), fmt='o',color='g',label='reduce_chisq')
    plt.legend()
    plt.xlabel('reduce chi square')
    plt.ylabel('Counts')
    plt.savefig(f"{output_dir}/reduce_chisq_hist.png")
    plt.clf()

   



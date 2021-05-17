import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
import math
import glob
import copy

"""
This script makes a graph of projected brightness and superimposed one. 
input: Gd9micron_making_projection/*.json
input: Gd9micron_making_superimposed_graph/*.png; two types of graphs.

cosole out example:
Original_0_00000000_0_8.png.json: left_edge = 47 pixel, 2.585 micron / right_edge = 42 pixel, 2.310 micron
Original_0_00000001_0_17.png.json: left_edge = 49 pixel, 2.695 micron / right_edge = 41 pixel, 2.255 micron
Original_0_00000002_0_15.png.json: left_edge = 47 pixel, 2.585 micron / right_edge = 39 pixel, 2.145 micron
Original_0_00000003_0_19.png.json: left_edge = 43 pixel, 2.365 micron / right_edge = 41 pixel, 2.255 micron
Original_0_00000004_0_18.png.json: left_edge = 44 pixel, 2.420 micron / right_edge = 42 pixel, 2.310 micron
Original_0_00000005_0_17.png.json: left_edge = 47 pixel, 2.585 micron / right_edge = 44 pixel, 2.420 micron
Original_0_00000006_0_17.png.json: left_edge = 48 pixel, 2.640 micron / right_edge = 40 pixel, 2.200 micron
Original_0_00000007_0_18.png.json: left_edge = 54 pixel, 2.969 micron / right_edge = 36 pixel, 1.980 micron
"""


if __name__ == '__main__':

    # output
    output_dir = "Gd9micron_making_superimposed_graph"
    os.makedirs(f'{output_dir}', exist_ok=True)

    pix_to_micron = 0.05499# 2020.02.28 HENPneutronandemulsionpresentation.pdf

    list_json_file = glob.glob("Gd9micron_making_projection/*.json")
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


        fig = plt.figure(figsize=(6.0, 6.0))
        #
        ax = fig.add_subplot(1, 1, 1)
        graph_x = np.arange(0, len(file_info['array_projection']))
        ax.plot(graph_x, file_info['array_projection'], marker='o', c = 'blue',linewidth=6)
        for p in list_peak_x:
            ax.axvline(p, c = 'r')
        #ax.set_ylim(0, 255)
        ax.set_title("Y projection")
        ax.set_xlabel("Y")
        ax.set_ylabel("average brightness")
        #
        fig.tight_layout()
        plt.savefig(f"{output_dir}/{base_name}_peaks.png")
        #plt.show()
        plt.clf()




        list_superimposed_val = [0 for i in range(int(lam))]
        for j in range(len(list_peak_x)-1):
        #for j in range(1):
            for i in range(int(lam)):
                x_pointer = int(list_peak_x[j] + i)
                if x_pointer >= len(file_info['array_projection']):
                    break
                list_superimposed_val[i] += file_info['array_projection'][x_pointer]


        left_height = abs(list_superimposed_val[0] - min(list_superimposed_val))
        left_10 = list_superimposed_val[0] - left_height * 0.10
        left_90 = list_superimposed_val[0] - left_height * 0.90
        #
        left_x_10 = 0
        left_x_90 = 0
        for i, v in enumerate(list_superimposed_val):
            if v < left_10:
                left_x_10  = i
                break
        for i, v in enumerate(list_superimposed_val):
            if v < left_90:
                left_x_90  = i
                break
        left_edge = left_x_90 - left_x_10
        print(f"{base_name}: left_edge = {left_edge} pixel, {left_edge * pix_to_micron:.3f} micron / ", end="")



        right_height = abs(list_superimposed_val[-1] - min(list_superimposed_val))
        right_10 = list_superimposed_val[-1] - right_height * 0.10
        right_90 = list_superimposed_val[-1] - right_height * 0.90
        right_x_10 = 0
        right_x_90 = 0
        for i, v in enumerate(reversed(list_superimposed_val)):
            if v < right_10:
                right_x_10  = i
                break
        for i, v in enumerate(reversed(list_superimposed_val)):
            if v < right_90:
                right_x_90  = i
                break
        right_edge = right_x_90 - right_x_10
        print(f"right_edge = {right_edge} pixel, {right_edge * pix_to_micron:.3f} micron")


        #
        graph_x = np.arange(0, len(list_superimposed_val))
        fig = plt.figure(figsize=(12.0, 6.0))
        #
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(graph_x, list_superimposed_val, c = 'blue', linewidth=6)
        ax.axhline(min(list_superimposed_val),c= 'r')
        ax.axhline(left_10, c= 'r')
        ax.axhline(left_90, c= 'r')
        ax.axhline(list_superimposed_val[0], c= 'r')
        ax.set_title("Brightness")
        ax.set_xlabel("phase [pixel]")
        ax.set_ylabel("brightness")
        ax.text(len(list_superimposed_val)/4, left_90, f"{left_edge:.2f} pixel, {left_edge * pix_to_micron:.2f} micron", size=18)


        #
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(graph_x, list_superimposed_val, c = 'blue', linewidth=6)
        ax.axhline(min(list_superimposed_val), c= 'r')
        ax.axhline(right_10, c= 'r')
        ax.axhline(right_90, c= 'r')
        ax.axhline(list_superimposed_val[-1], c= 'r')
        ax.set_title("Brightness")
        ax.set_xlabel("phase [pixel]")
        ax.set_ylabel("brightness")
        ax.text(len(list_superimposed_val)/4, right_90, f"{right_edge:.2f} pixel, {right_edge * pix_to_micron:.2f} micron", size=18)

        #
        fig.tight_layout()
        plt.savefig(f"{output_dir}/{base_name}_riging_edge.png")
        #plt.show()
        plt.clf()





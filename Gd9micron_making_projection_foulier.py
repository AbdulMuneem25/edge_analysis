import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
import math
import glob
import copy

from scipy import optimize


"""
This script makes numerical list of projected brightness and sinecurve fitting as a json file per view.
The parameters obtained by this fitting (in particular, phi) will be used to generate superimpose images. 
input: images/div_images_trial17/*.png
output: {output_dir}/{file_basename}.json
"""



if __name__ == '__main__':

    # params
    pix_to_micron = 0.05499# 2020.02.28 HENPneutronandemulsionpresentation.pdf
    lam = 9.0 / 0.05499 # lambda is a fixed value.
    # output
    output_dir = "Gd9micron_making_projection_foulier"
    os.makedirs(f'{output_dir}', exist_ok=True)

    # input
    input_root_dir = "images"
    dataset_dir = "div_images_trial17"
    list_file = glob.glob(f"{input_root_dir}/{dataset_dir}/*.png")


    for file_name in list_file:
        file_basename = os.path.basename(file_name)

        img_src = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        height = img_src.shape[0]
        width = img_src.shape[1]
        center_y = int(height/2)
        center_x = int(width/2)
        y0 = center_y - int(height/4)
        x0 = center_x - int(height/4)
        y1 = center_y + int(height/4)
        x1 = center_x + int(height/4)
        img_rotated_roi = img_src[y0:y1,x0:x1]
        #cv2.imshow("img_rotated_roi", img_rotated_roi)
        #cv2.waitKey()
        cv2.imwrite(f"{output_dir}/{file_basename}_cropped_GDsection.png", img_rotated_roi)

        # projection
        reduced_vec_1 = cv2.reduce(img_rotated_roi, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).T           

        # foulier trans
        array_x = np.arange(0, len(reduced_vec_1[0]))
        cos_curve = np.cos(array_x * 2 * np.pi / lam )
        sin_curve = np.sin(array_x * 2 * np.pi / lam )
        real_inner_product = np.dot(reduced_vec_1, cos_curve)
        imag_inner_product = np.dot(reduced_vec_1, sin_curve)
        absolute_val = math.hypot(real_inner_product, imag_inner_product)
        angle = np.arctan2(imag_inner_product, real_inner_product)
        phase = angle / (2.0 * np.pi) * lam

        # graph
        fig = plt.figure(figsize=(12.0, 6.0))
        #
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(img_src)
        ax.set_title(f"{file_basename}")
        #
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(array_x, reduced_vec_1[0], marker='o')
        lines = []
        line_peak = phase
        lines.append(line_peak.tolist())
        while line_peak < max(array_x):
            ax.axvline(phase)
            phase += lam / 2.0
            lines.append(phase.tolist())
        ax.set_title("Y projection")
        ax.set_xlabel("Y")
        ax.set_ylabel("Brightness sum")
        plt.savefig(f"{output_dir}/{file_basename}.png")
        plt.clf()

        list_lines = [int(num) for nums in lines for num in nums]


        my_dict = {"file_name": f"{file_name}",
                   "array_projection": reduced_vec_1[0].tolist(),
                   "phase": list_lines,
                  }
        with open(f'{output_dir}/{file_basename}.json', 'w') as jsonfile:
            json.dump(my_dict, jsonfile, indent=4)





import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
import math
import glob
import copy

from scipy import optimize



if __name__ == '__main__':

    # output
    output_dir = "Gd9micron_thickness_projection"
    os.makedirs(f'{output_dir}', exist_ok=True)


    img_src = cv2.imread("images/Gd_pattern_cross_section.png", cv2.IMREAD_GRAYSCALE)


    for ksize in np.arange(5, 105, 10):
        img = copy.deepcopy(img_src)
        img = cv2.GaussianBlur(img, (ksize, 1), -1)
        reduced_vec = cv2.reduce(img, 0, cv2.REDUCE_AVG)
        graph_x = np.arange(0, len(reduced_vec[0]))
        #
        fig = plt.figure(figsize=(6.0, 12.0))
        #
        ax = fig.add_subplot(2, 1, 1)
        ax.imshow(img)
        #
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(graph_x, reduced_vec[0])
        ax.set_ylim(0, 255)
        ax.set_title("Thickness of Gd, Projection")
        ax.set_xlabel("X")
        ax.set_ylabel("average brightness")
        #
        fig.tight_layout()
        plt.savefig(f"{output_dir}/ksize_{ksize:03d}.png")
        #plt.show()
        plt.clf()


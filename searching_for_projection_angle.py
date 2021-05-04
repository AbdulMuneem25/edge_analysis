import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
import math
import glob
import copy

from scipy import optimize


def fitting_pol2_func(x, a, b, c):
    return a * (x - b)**2 + c


if __name__ == '__main__':

    output_dir = "searching_for_projection_angle"
    os.makedirs(f'{output_dir}', exist_ok=True)

    # input_dir = "000_sin_curve_1024"
    #input_dir = "real_images_trial17"
    input_dir = "making_test_pattern_for_MTF"

    list_file = glob.glob(f'images/{input_dir}/*.png')
    list_best_angle = []

    for f in list_file:
        filename = os.path.basename(f)
        img_src = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        height = img_src.shape[0]
        width = img_src.shape[1]
        center_y = int(height/2)
        center_x = int(width/2)

        list_rotate_deg = np.arange(-1, 1, 0.1)
        list_std = []
        for i, rotate_deg in enumerate(list_rotate_deg):

            print(rotate_deg)
            rotate = np.deg2rad(rotate_deg)
            scale = 1.0
            rotation_matrix  = cv2.getRotationMatrix2D((center_x, center_y), rotate_deg, scale)
            img_rotated = cv2.warpAffine(img_src, rotation_matrix, img_src.shape, flags=cv2.INTER_CUBIC)
            # cv2.imshow("img_rotated", img_rotated)
            # cv2.waitKey()

            #
            reduced_vec_0 = cv2.reduce(img_rotated, 0, cv2.REDUCE_AVG)
            reduced_vec_1 = cv2.reduce(img_rotated, 1, cv2.REDUCE_AVG).T           
            reduced_vec_0_copy = copy.deepcopy(reduced_vec_0)
            reduced_vec_1_copy = copy.deepcopy(reduced_vec_1)
            graph_x = np.arange(0, len(reduced_vec_1[0]))

            reduced_vec_0_copy[0][0:100] = 0
            reduced_vec_0_copy[0][-100:] = 0
            reduced_vec_1_copy[0][0:100] = 0
            reduced_vec_1_copy[0][-100:] = 0

            this_std = np.std(reduced_vec_1[0][100:-100])
            this_mean = np.mean(reduced_vec_1[0][100:-100])
            list_std.append(this_std)

            #
            fig = plt.figure()
            #
            ax = fig.add_subplot(2, 2, 1)
            ax.imshow(img_rotated)
            ax.set_title(f"image rotation = {rotate_deg:.2f}deg")
            #
            ax = fig.add_subplot(2, 2, 3)
            ax.plot(graph_x, reduced_vec_1[0])
            ax.plot(graph_x, reduced_vec_1_copy[0])
            ax.set_ylim(0, 255)
            ax.set_title("Y projection")
            ax.set_xlabel("Y")
            ax.set_ylabel("average brightness")
            ax.text(len(reduced_vec_1[0])/4, 0, f"{this_mean:.2f} +/- {this_std:.2f}", size=10)
            #
            ax = fig.add_subplot(2, 2, 4)
            ax.plot(graph_x, reduced_vec_0[0])
            ax.plot(graph_x, reduced_vec_0_copy[0])
            ax.set_ylim(0, 255)
            ax.set_title("X projection")
            ax.set_xlabel("X")
            ax.set_ylabel("average brightness")
            #
            fig.tight_layout()
            plt.savefig(f"{output_dir}/{filename}_i{i:02d}_rotate{rotate_deg:.2f}.png")
            #plt.show()
            plt.clf()


        my_initial_param = [-1, np.median(list_rotate_deg), max(list_std)]
        params, params_covariance = optimize.curve_fit(fitting_pol2_func, list_rotate_deg, list_std, p0=my_initial_param)
        list_best_angle.append(params[1])


        ax = fig.add_subplot(1, 1, 1)
        ax.plot(list_rotate_deg, list_std, marker='o')
        ax.plot(list_rotate_deg, fitting_pol2_func(list_rotate_deg, params[0], params[1], params[2]),)
        ax.set_title("STD as a function of rotation")
        ax.set_ylabel("STD")
        ax.set_xlabel("rotation angle [deg]")
        ax.text(params[1], params[2], f"deg={params[1]:.3f} +/- {params_covariance[1][1]:.3f}", size=10)

        plt.savefig(f"{output_dir}/{filename}_fitting.png")
        print(params_covariance)
        plt.clf()

    list_dict = []
    for filename, angle in zip(list_file, list_best_angle):
        this_dict = {"file_name":filename, "rotation_angle":angle}
        list_dict.append(this_dict)

    with open(f'{output_dir}/{input_dir}.json', 'w') as jsonfile:
        json.dump(list_dict, jsonfile, indent=4)


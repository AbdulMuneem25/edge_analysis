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


def fitting_sin_func(x, amp, phi, bias):
    lam = 9.0 / 0.05499 # lambda is a fixed value.
    return amp * np.sin( 2 * np.pi * x / lam + phi) + bias



if __name__ == '__main__':

    # params
    pix_to_micron = 0.05499# 2020.02.28 HENPneutronandemulsionpresentation.pdf
    # output
    output_dir = "Gd9micron_making_projection"
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


        reduced_vec_1 = cv2.reduce(img_rotated_roi, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).T  
        print('reduced_vec_1[0]',reduced_vec_1[0])  
        graph_x = np.arange(0, len(reduced_vec_1[0]))

        init_amp = (np.max(reduced_vec_1) - np.min(reduced_vec_1)) / 2.0
        my_initial_params = [init_amp, 0.0, np.mean(reduced_vec_1)]
        params, params_covariance = optimize.curve_fit(fitting_sin_func, graph_x, reduced_vec_1[0], p0=my_initial_params)

        print('params[0]',params[0])
        print('params[1]',params[1])
        print('params[2]',params[2])


        #
        fig = plt.figure(figsize=(12.0, 6.0))
        #
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(img_src)
        ax.set_title(f"{file_basename}")
        #
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(graph_x, reduced_vec_1[0], marker='o')
        ax.plot(graph_x, fitting_sin_func(graph_x, params[0], params[1], params[2]),)
        ax.text(len(reduced_vec_1[0])/4, 0, f"{params[1]:.2f} +/- {params_covariance[1][1]:.2f}", size=10)
        #print(f"{file_name}: {params[1] * pix_to_micron:.3f} +/- {params_covariance[1][1] * pix_to_micron:.3f}, ", end="")
        #print(f"{params[1]:.2f} +/- {params_covariance[1][1]:.2f}")
        #ax.set_ylim(0, 255)
        ax.set_title("Y projection")
        ax.set_xlabel("Y")
        ax.set_ylabel("average brightness")
        #
        #fig.tight_layout()
        plt.savefig(f"{output_dir}/{file_basename.replace('.png', '')}.png")
        #plt.show()
        plt.clf()

        my_dict = {"file_name": f"{file_name}",
                   "array_projection": reduced_vec_1[0].tolist(),
                   "fitting_amplitude": params[0],
                   "fitting_phi": params[1],
                   "fitting_bias": params[2],
                   "fitting_amplitude_error": params_covariance[0][0],
                   "fitting_phi_error": params_covariance[1][1],
                   "fitting_bias_error": params_covariance[2][2]
                  }
        with open(f"{output_dir}/{file_basename.replace('.png', '')}.json", 'w') as jsonfile:
            json.dump(my_dict, jsonfile, indent=4)
        with open(f"{output_dir}/{file_basename.replace('.png', '')}.txt", 'w') as f:
            for h, g in zip(graph_x, reduced_vec_1[0]):
                f.writelines("{} {}\n".format(h, g))











        # graph = ROOT.TGraphErrors()
        # for i in range(len(graph_x)):
        #     graph.SetPoint(i, graph_x[i], reduced_vec_1[0][i])
        #     #graph.SetPointError(i, yerr[i], yerr[i])
        # func = ROOT.TF1("Name", "gaus")
        # graph.Fit(func)

        # canvas = ROOT.TCanvas("name", "title", 1024, 768)
        # graph.GetXaxis().SetTitle("x") # set x-axis title
        # graph.GetYaxis().SetTitle("y") # set y-axis title
        # graph.Draw("AP")

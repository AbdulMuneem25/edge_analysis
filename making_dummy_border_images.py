import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
import math


if __name__ == '__main__':

    output_dir = "making_dummy_border_images"
    os.makedirs(f'{output_dir}', exist_ok=True)


    width = 1024
    height = 1024
    max_brightness = 255
    img = np.zeros((height, width), dtype=np.uint8)


    list_rotate_deg = [0.0, 0.1, 0.2, 0.5, 1.0, -0.5]
    list_period = [10, 12]
    binary_threshold = 100

    for rotate_deg in list_rotate_deg:
        for period in list_period:
            rotate = np.deg2rad(rotate_deg)

            for y in range(height):
                for x in range(width):
                    x0 = x * np.cos(rotate) + y * np.sin(rotate)
                    y0 = -x * np.sin(rotate) + y * np.cos(rotate)
                    amplitude  = max_brightness / 2.0
                    phase = y0 / height * 2 * np.pi * period 
                    bias = max_brightness / 2.0
                    brightness =  amplitude * np.cos(phase) + bias
                    brightness_int = int(brightness)
                    img[y, x] = brightness_int
    
            #_, img = cv2.threshold(img, binary_threshold, 255, type=cv2.THRESH_TOZERO)
            # cv2.imshow("img", img)
            # cv2.waitKey()
            cv2.imwrite(f"{output_dir}/period{period}_rotate{rotate_deg:.2f}.png", img)
        

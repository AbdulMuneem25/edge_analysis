import numpy as np
import cv2
import os


if __name__ == '__main__':

    output_dir = "images/making_test_pattern_for_MTF"
    os.makedirs(f'{output_dir}', exist_ok=True)

    width = 512
    height = 512
    list_line_interval = [10, 8, 5, 4, 3, 2, 1]


    for interval in list_line_interval:
        y_pointer = 0
        print(interval)
        img = np.ones((height, width), dtype=np.uint8) * 255

        while y_pointer < height:
            x0 = 0
            x1 = width
            y0 = y_pointer
            y1 = y_pointer + interval - 1
            img = cv2.rectangle(img, (x0, y0), (x1, y1), 0, thickness=-1)
            y_pointer += interval * 2

        cv2.imwrite(f"{output_dir}/interval{interval:02d}_gaussianblur00.png", img)

        for kernel_size in np.arange(3, 21, 2):
            img_blured = cv2.blur(img, (kernel_size, kernel_size))
            cv2.imwrite(f"{output_dir}/interval{interval:02d}_gaussianblur{kernel_size:02d}.png", img_blured)

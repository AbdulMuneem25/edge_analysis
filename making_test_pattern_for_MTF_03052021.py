import numpy as np
import cv2
import os


if __name__ == '__main__':

    output_dir = "making_test_pattern_for_MTF"
    os.makedirs(f'{output_dir}', exist_ok=True)

    width = 1024
    height = 1024
    n_each_lines = 12
    list_line_interval = [10, 8, 5, 4, 3, 2, 1]

    y_pointer = 0
    img = np.ones((height, width), dtype=np.uint8) * 255
    for interval in list_line_interval:
        n_each_lines = int( height / (len(list_line_interval)*2) / interval)
        print(f"interval={interval}, start_y={y_pointer},", end="")
        for i in range(n_each_lines):
            x0 = 0
            x1 = width
            y0 = y_pointer
            y1 = y_pointer + interval - 1
            img = cv2.rectangle(img, (x0, y0), (x1, y1), 0, thickness=-1)
            y_pointer += interval * 2

        y_pointer += interval * 2
        print(f"end_y={y_pointer}")

        # cv2.imshow("img", img)
        # cv2.waitKey()


    cv2.imwrite(f"{output_dir}/gaussianblur_00.png", img)

    for kernel_size in np.arange(3, 21, 2):
        img_blured = cv2.blur(img, (kernel_size, kernel_size))
        cv2.imwrite(f"{output_dir}/gaussianblur_{kernel_size:02d}.png", img_blured)

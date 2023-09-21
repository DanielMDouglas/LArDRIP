import os
import cv2
import numpy as np


def std_mean(folder_path):
    channel_means = []
    channel_stds = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # I think we will only use .png's,but just in case:
        if file_path.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            # Read the image using OpenCV
            image = cv2.imread(file_path)

            # Calculate mean for each channel separately
            channel_means.append(np.mean(image, axis=(0, 1)))
            channel_stds.append(np.std(image, axis=(0, 1)))

    channel_means = np.array(channel_means)
    channel_stds = np.array(channel_stds)

    # find the avg of each channel, should return something like [number, number, number] for each
    avg_means = np.mean(channel_means, axis=0)
    avg_stds = np.mean(channel_stds, axis=0)

    return [avg_means.tolist(), avg_stds.tolist()]


###
# test:
# folder_path = r"C:\Users\Hilary\Desktop\mae_main_new\LArDRIP\train"
# print(std_mean(folder_path))
###

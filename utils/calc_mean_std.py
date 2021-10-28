from PIL import Image
import numpy as np
import os
import glob


def get_mean_std(path):

    mean_sum = 0
    std_sum = 0
    n_imgs = 0

    for i, image in enumerate(glob.glob(f"{path}/*")):

        image = Image.open(image).convert('L')
        data = np.asarray(image)
        
        mean = np.mean(data)
        std = np.std(data)

        mean_sum += mean
        std_sum += std
        n_imgs = i+1

    mean = mean_sum/(n_imgs)
    std = std_sum/(n_imgs)

    print("mean: ", mean, "std: ", std)

    return mean, std

if __name__ == "__main__":

    path = r"G:\My Drive\Projektarbeit_ResearchProject\datasets\BUS_classification\resized\train_val\256\500\images"

    get_mean_std(path)
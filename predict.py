import torch

import numpy as np
import os
import pandas as pd
import albumentations
import argparse

from utils.data import ClassificationDataset


def load_data(test_data_path, gt):

    df_test = pd.read_csv(gt)
    mean = 0.1685
    std = 0.1796

    test_images = df_test.image.values.tolist()
    test_images = [os.path.join(test_data_path, i) for i in test_images]
    test_targets = df_test.target.values

    test_aug = albumentations.Compose(
            [
                albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
            ]
        )

    train_dataset = ClassificationDataset(
            image_paths=test_images,
            targets=test_targets,
            augmentations=test_aug)


def predict(model, device, test_loader, loss_function):

    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):

            data= data.to_(device)
            output = model(data)

            output = torch.sigmoid(output)
            output = output.cpu().numpy()

            for j in range(len(output)):
                if output[j] > 0.5:
                    output[j] = 1
                else:
                    output[j] = 0

            if i==0:
                predictions = output
            else: 
                predictions = np.concatenate(predictions,output)

    return predictions


def load_model():


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    
    parser.add_argument("--dataset", type=str, help="path to the folder containing the images")
    parser.add_argument("--gt", type=str, help="path to the file that contains the gt csv file - see examples under /examples (needs to be added)")
    parser.add_argument("--outdir", type=str, help="outputdir where the files are stored")

    opt = parser.parse_args()

    load_data()
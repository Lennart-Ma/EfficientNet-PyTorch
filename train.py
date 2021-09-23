import os
from sklearn import metrics
import torch
from torch import nn

import albumentations
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import json
import matplotlib.pyplot as plt

from efficientnet_pytorch import EfficientNet
from utils.data import ClassificationDataset
from utils.training_loops import training_loop
from utils.training_loops import val_loop

def train(fold, training_data_path, gt, device, epochs, train_bs, val_bs, outdir, lr):

    df = pd.read_csv(gt)
    mean = 0.1685
    std = 0.1796

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_val = df[df.kfold == fold].reset_index(drop=True)

    model = EfficientNet.from_pretrained("efficientnet-b2", in_channels = 1, num_classes = 1)
    model.to(device)

    # Set up the train_loader and val_loader
    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            albumentations.Flip(p=0.5)
        ]
    )

    val_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    train_images = df_train.image.values.tolist()
    train_images = [os.path.join(training_data_path, i) for i in train_images]
    train_targets = df_train.target.values

    val_images = df_val.image.values.tolist()
    val_images = [os.path.join(training_data_path, i) for i in val_images]
    val_targets = df_val.target.values

    train_dataset = ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        augmentations=train_aug)

    val_dataset = ClassificationDataset(
        image_paths=val_images,
        targets=val_targets,
        augmentations=val_aug)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=2
    )

    # Set loss function
    loss_function = nn.BCEWithLogitsLoss()

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        threshold=0.0001,
        mode="max"
    )

    train_loss_all = []
    accuracy_list = []
    val_loss_list = []
    prev_accuracy = 0
    # Start training
    for epoch in range(epochs):

        train_loss = training_loop(model, device, train_loader, optimizer, loss_function, epoch, epochs)

        targets, predictions, accuracy, val_loss = val_loop(model, device, val_loader, loss_function)

        assert np.array_equal(targets, val_targets), "targets from val_loop are not equal to val_targets (source of the validation data)"

        predictions = np.vstack((predictions)).ravel()

        auc = metrics.roc_auc_score(targets, predictions)

        print(f"Epoch = {epoch+1}, AUC = {auc}")
    
        scheduler.step(auc)
    
        if accuracy > prev_accuracy:
            torch.save(model.state_dict(), os.path.join(outdir, f"model_fold_{fold}.bin"))
            print("Better model saved to outdir")

        prev_accuracy = accuracy

        train_loss_all.append(train_loss)
        val_loss_list.append(val_loss)
        accuracy_list.append(accuracy)


    # Make a function for the following lines
    train_loss_all = np.array(train_loss_all)
    train_loss_all = train_loss_all.flatten()
    train_loss_plot = plt.figure()
    plt.plot(train_loss_all)
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    train_loss_plot.savefig(os.path.join(opt.outdir, f'training_loss_fold{opt.fold}.png'))

    val_loss_plot = plt.figure()
    plt.plot(val_loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    val_loss_plot.savefig(os.path.join(opt.outdir, f'validation_loss_fold{opt.fold}.png'))

    accuracy_loss_plot = plt.figure()
    plt.plot(accuracy_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    accuracy_loss_plot.savefig(os.path.join(opt.outdir, f'accuracy_fold{opt.fold}.png'))
        


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, help="which fold is the val fold")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--train_batch", type=int, default=64, help="batch size for training")
    parser.add_argument("--val_batch", type=int, default=64, help="batch size for validation")
    parser.add_argument("--lr", type=int, default=1e-4)
    parser.add_argument("--dataset", type=str, help="path to the folder containing the images")
    parser.add_argument("--gt", type=str, help="path to the file that contains the gt csv file - see examples under /examples (needs to be added)")
    parser.add_argument("--outdir", type=str, help="outputdir where the files are stored")

    opt = parser.parse_args()

    opt_dict = vars(opt)

    json_object = json.dumps(opt_dict)

    open(os.path.join(opt.outdir, f'training_options{opt.fold}.txt'), 'w').close()

    with open(os.path.join(opt.outdir, f'training_options{opt.fold}.txt'), 'a') as f:
        json.dump(json_object, f)
    
    train(opt.fold, opt.dataset, opt.gt, opt.device, opt.epochs, opt.train_batch, opt.val_batch, opt.outdir, opt.lr)
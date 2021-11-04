import torch

import numpy as np
import os
import pandas as pd
import albumentations
import argparse
import glob
import csv

from efficientnet_pytorch import EfficientNet
from utils.data import ClassificationDataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from utils.calc_mean_std import get_mean_std


def load_data(test_data_path, gt):

    df_test = pd.read_csv(gt)

    test_images = df_test.image.values.tolist()
    test_images = [os.path.join(test_data_path, i) for i in test_images]
    test_targets = df_test.target.values

    return test_images, test_targets


def predict(model, device, test_loader, ensemble):

    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):

            target = target.type(torch.DoubleTensor)
            data= data.to(device)
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
                targets = target.data.cpu().numpy()
            else: 
                predictions = np.concatenate((predictions,output))
                targets = np.concatenate((targets, target.data.cpu().numpy()))

    predictions = predictions.flatten()

    if ensemble:
        return targets, predictions
    
    accuracy = accuracy_score(targets, predictions)
    auc = roc_auc_score(targets, predictions)

    print(predictions)
    print()
    print()

    return classification_report(targets, predictions, output_dict=True), accuracy, auc


def calc_ensemble_prediction(collected_predictions):

    """
    Returns the average prediction of all three model's predictions - example see below
    
    targets: numpy array of the y_true values of the test_data
    collected_predictions: list (of length 3) of numpy arrays, where each numpy array is the y_predict of the corresponding model_{fold} 

    model0 = [1., 1., 1., 1., 0.]
    model1 = [0., 1., 1., 0., 0.]
    model2 = [1., 1., 0., 0., 0.]
    ensembled_prediction = [2.0, 3.0, 2.0, 1.0, 0.0]
    ensembled_predictions_one_hot = [1. 1. 1. 0. 0.] --> return
    """

    ensembled_predictions_one_hot = []
    ensembled_predictions= [0.] * len(collected_predictions[0])

    for i in range(len(collected_predictions)):
        for j in range(len(collected_predictions[i])):
            ensembled_predictions[j] += collected_predictions[i][j]

    for i in range(len(ensembled_predictions)):

        if int(ensembled_predictions[i]) >= 2:
            ensembled_predictions_one_hot.append(1.0)
        else:
            ensembled_predictions_one_hot.append(0.0)

    return np.array(ensembled_predictions_one_hot)

    


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", type=str, help="path to the folder where the results (.csv) should be saved")
    parser.add_argument("--dataset", type=str, help="path to the folder containing the images size: 260x260")
    parser.add_argument("--gt", type=str, help="path to the file that contains the gt csv file - see examples under /examples (needs to be added)")
    parser.add_argument("--perc_data", type=int, help="How much data is used for classification in percentage (Used to name the output csv file uniquely")
    parser.add_argument("--fake", type=str, help="Use to name the output csv file - if fake data is used type 'fake' else 'no_fake'")


    parser.add_argument("--single_model_path", default=None, type=str, help="Use only for single model prediction: path to the folder containing the model file: .bin")
    parser.add_argument("--ensemble", type=bool, default=False, help="set to True for ensemble prediction")
    parser.add_argument("--Model0", type=str, default=None, help="Use only for ensemble model prediction: path to the folder containing the 1st model file for ensemble: .bin")
    parser.add_argument("--Model1", type=str, default=None, help="Use only for ensemble model prediction: path to the folder containing the 2nd model file for ensemble: .bin")
    parser.add_argument("--Model2", type=str, default=None, help="Use only for ensemble model prediction: path to the folder containing the 3rd model file for ensemble: .bin")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val_bs", type=int, default=16)
    parser.add_argument("--mean", type=float, default=None)
    parser.add_argument("--std", type=float, default=None)


    opt = parser.parse_args()
    
    test_images, test_targets = load_data(opt.dataset, opt.gt)

    if opt.mean is None and opt.std is None:
        print("Mean and std get calculated.. This may take 5 mins")
        mean, std = get_mean_std(opt.dataset)
    else:
        print("Mean and std are taken from the input - If you want to let it be calculated leave the args mean and std as None")
        mean = opt.mean
        std = opt.std
        print("mean: ", mean, "std: ", std)

    test_aug = albumentations.Compose(
            [
                albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
            ]
        )

    train_dataset = ClassificationDataset(
            image_paths=test_images,
            targets=test_targets,
            augmentations=test_aug)

    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.val_bs,
        shuffle=False,
        num_workers=2
    )


    if opt.ensemble:

        ensemble_list = [opt.Model0, opt.Model1, opt.Model2]
        collected_predictions = []
        collected_targets = []
        
        for i in range(len(ensemble_list)):

            model = EfficientNet.from_name("efficientnet-b2", in_channels = 1, num_classes = 1)
            model.load_state_dict(torch.load(ensemble_list[i]))
            model.to(opt.device)
            targets, predictions = predict(model, opt.device, test_loader, ensemble=True)
            
            collected_predictions.append(predictions)
            collected_targets.append(targets)

        assert np.array_equal(collected_targets[0].all(), collected_targets[1].all(), collected_targets[2].all()), "targets are not equal" #only possible if we have 3 models (3 fold)

        ensemble_pred = calc_ensemble_prediction(collected_predictions)

        accuracy = accuracy_score(targets, ensemble_pred)

        auc = roc_auc_score(targets, ensemble_pred)

        classification_report_ensemble = classification_report(targets, ensemble_pred, output_dict=True)

        print("Ensemble results: ")
        print(classification_report_ensemble)
        print("Accuracy: ", accuracy)
        print("AUC: ", auc)

        file_exist=False
        perc_data = opt.perc_data
        fake = opt.fake

        for file in glob.glob(os.path.join(opt.output_path, '*.{}'.format("csv"))):
            file_exist = True
            with open(file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["Ensemble results: "])
                writer.writerow(["label 1"])
                writer.writerow([classification_report_ensemble[list(classification_report_ensemble.keys())[0]]])
                writer.writerow(["label 2"])
                writer.writerow([classification_report_ensemble[list(classification_report_ensemble.keys())[1]]])
                writer.writerow(["Accuracy: ", accuracy])
                writer.writerow(["AUC: ", auc])
                writer.writerow([""])
        
        if not file_exist:
            with open(os.path.join(opt.output_path, f"Prediction_results_on_test_data_IVUS{perc_data}+{fake}.csv"), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([f"PREDICTION RESULTS ON TEST DATA WITH EfficientNetb2 model trained on IVUS data {perc_data}+{fake}"])
                writer.writerow([""])
                writer.writerow(["Ensemble results: "])
                writer.writerow(["label 1"])
                writer.writerow([classification_report_ensemble[list(classification_report_ensemble.keys())[0]]])
                writer.writerow(["label 2"])
                writer.writerow([classification_report_ensemble[list(classification_report_ensemble.keys())[1]]])
                writer.writerow(["Accuracy: ", accuracy])
                writer.writerow(["AUC: ", auc])
                writer.writerow([""])

    elif opt.single_model_path is not None:
        
        model = EfficientNet.from_name("efficientnet-b2", in_channels = 1, num_classes = 1)
        model.load_state_dict(torch.load(opt.single_model_path))
        model.to(opt.device)

        single_report, accuracy, auc = predict(model, opt.device, test_loader, ensemble=False)

        print("Single Model results: ", single_report)
        print("Accuracy: ", accuracy)
        print("AUC: ", auc)

        file_exist=False
        perc_data = opt.perc_data
        fake = opt.fake

        for file in glob.glob(os.path.join(opt.output_path, '*.{}'.format("csv"))):
            file_exist = True
            with open(file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["Results for : ", os.path.basename(opt.single_model_path)])
                writer.writerow(["label 1"])
                writer.writerow([single_report[list(single_report.keys())[0]]])
                writer.writerow(["label 2"])
                writer.writerow([single_report[list(single_report.keys())[1]]])
                writer.writerow(["Accuracy: ", accuracy])
                writer.writerow(["AUC: ", auc])
                writer.writerow([""])
        
        if not file_exist:
            with open(os.path.join(opt.output_path, f"Prediction_results_on_test_data_IVUS{perc_data}+{fake}.csv"), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([f"PREDICTION RESULTS ON TEST DATA WITH EfficientNetb2 model trained on IVUS data {perc_data}+{fake}"])
                writer.writerow([""])
                writer.writerow(["Results for : ", os.path.basename(opt.single_model_path)])
                writer.writerow(["label 1"])
                writer.writerow([single_report[list(single_report.keys())[0]]])
                writer.writerow(["label 2"])
                writer.writerow([single_report[list(single_report.keys())[1]]])
                writer.writerow(["Accuracy: ", accuracy])
                writer.writerow(["AUC: ", auc])
                writer.writerow([""])

    else:
        print("No model path specified - read opt.arguments")

import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold 


def export_hist(path, gt_path, name):
    """
    plots the histogram by the different folds of the gt.csv of given path 
    
    path: path to the csv with kfold as column
    name: name of the .png which gets exported 
    """
    gt = pd.read_csv(gt_path)

    gt.hist("target", bins=2, grid=False, by="kfold")
    plt.savefig(os.path.join(path, f'{name}.png'))

def create_stratified_gt(old_gt_path, new_gt_path, folds):

    """
    Creates a new ground truth csv file with a stratified kfold column

    old_gt_path: path to the gt csv file with non stratified kfolds
    new_gt_path: path to the new csv file with stratified kfolds
    folds: number of folds which you want to have for training (cross validation) 
    """

    gt = pd.read_csv(old_gt_path)

    gt = gt.drop(columns="kfold")
    gt["kfold"] = "NaN"

    target_list = gt["target"].tolist()
    patient_list = gt["image"].tolist()
    skf = StratifiedKFold(n_splits=folds)
    skf.get_n_splits(patient_list, target_list)

    for i, (train_index, val_index) in enumerate(skf.split(patient_list, target_list)):
        for index in val_index:
            gt.loc[index, 'kfold'] = i

    gt.to_csv(new_gt_path, index=False)


if __name__ == "__main__":


    ## Part which have to be changed
    path = r"G:\My Drive\Projektarbeit_ResearchProject\datasets\IVUS\IVUS_resized\fakes\260\10"


    ## Change this if you rebuild fakes csv files
    gt_path = os.path.join(path, "train_merged.csv")

    gt_strat_path = os.path.join(path, "train_merged_stratified.csv")

    export_hist(path, gt_path, "hist_fold_distr_no_strat")

    create_stratified_gt(gt_path, gt_strat_path, 3)

    export_hist(path, gt_strat_path, "hist_fold_distr_strat")
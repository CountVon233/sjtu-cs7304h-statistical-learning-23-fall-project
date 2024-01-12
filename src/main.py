import numpy as np
import pickle
import scipy
from pathlib import Path
from IO import read_in 
from IO import write_out 
import run_PCA
import run_DL
import run_LR
import run_SVM

run_pca_flag = True
run_dl_flag = True
run_lr_flag = True
run_svm_flag = True

relative_train_feature_path = "..\\..\\dataset\\train_feature.pkl"
relative_train_label_path = "..\\..\\dataset\\train_labels.npy"
relative_test_feature_path = "..\\..\\dataset\\test_feature.pkl"
relative_pca_train_feature_path = "..\\..\\dataset\\pca_train_feature.npy"
relative_pca_test_feature_path = "..\\..\\dataset\\pca_test_feature.npy"

TrainFeature = read_in.read_file(relative_train_feature_path)
TrainLabel = read_in.read_file(relative_train_label_path)
TestFeature = read_in.read_file(relative_test_feature_path)
PCATrainFeature = read_in.read_file(relative_pca_train_feature_path)
PCATestFeature = read_in.read_file(relative_pca_test_feature_path)

if run_pca_flag :
    target_dim = 233
    run_PCA.run_pca(TrainFeature, TestFeature, target_dim)

if run_dl_flag :
    dl_result = run_DL.run_dl(PCATrainFeature, TrainLabel, PCATestFeature)
    dl_result_output_path = "..\\..\\output\\dl_out.csv"
    write_out.write_output(dl_result, dl_result_output_path)

if run_lr_flag :
    lr_result = run_LR.run_lr(PCATrainFeature, TrainLabel, PCATestFeature)
    lr_result_output_path = "..\\..\\output\\lr_out.csv"
    write_out.write_output(lr_result, lr_result_output_path)
    pass

if run_svm_flag :
    svm_result = run_SVM.run_svm(PCATrainFeature, TrainLabel, PCATestFeature)
    svm_result_output_path = "..\\..\\output\\svm_out.csv"
    write_out.write_output(svm_result, svm_result_output_path)
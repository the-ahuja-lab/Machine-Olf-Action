import MLPipeline
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from joblib import dump, load


DATA_FLD_NAME = "step4"
DATA_FILE_NAME_PRFX = "FE_"


class FeatureExtraction:

    def __init__(self, ml_pipeline: MLPipeline):
        print("Inside FeatureExtraction initialization")

        self.ml_pipeline = ml_pipeline

        if self.ml_pipeline.status == "feature_selection":  # resuming at step 4
            print(ml_pipeline.job_data)

            if self.ml_pipeline.data is None or self.ml_pipeline.x_train is None or self.ml_pipeline.x_test is None:
                padel_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step1", "FG_Padel.csv")
                train_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step3", "FS_train.csv")
                test_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step3", "FS_test.csv")

                # data, data_labels = self.read_data(padel_data_fp)
                # self.ml_pipeline.data = data
                # self.ml_pipeline.data_labels = data_labels

                x_train = pd.read_csv(train_data_fp)
                y_train = x_train['Activation Status']
                x_train = x_train.drop("Activation Status", axis=1)

                x_test = pd.read_csv(test_data_fp)
                y_test = x_test['Activation Status']
                x_test = x_test.drop("Activation Status", axis=1)

                self.ml_pipeline.x_train = x_train
                self.ml_pipeline.x_test = x_test
                self.ml_pipeline.y_train = y_train
                self.ml_pipeline.y_test = y_test

            self.perform_feature_extraction()

    def perform_feature_extraction(self):
        self.perform_pca()

        # TODO check if all above steps successful
        self.write_to_csv_and_update_status()

    def perform_pca(self):

        if self.ml_pipeline.config.fe_pca_flg:
            pca_energy = self.ml_pipeline.config.fe_pca_energy

            xtrain = self.ml_pipeline.x_train
            xtest = self.ml_pipeline.x_test

            print("Inside PCA, Before Shape Train: ", xtrain.shape)
            print("Inside PCA, Before Shape Test: ", xtest.shape)
            pca = PCA(pca_energy)
            pca.fit(xtrain)

            xtrain_new = pca.transform(xtrain)
            xtest_new = pca.transform(xtest)
            print("Inside PCA, After Shape Train: ", xtrain_new.shape)
            print("Inside PCA, After Shape Test: ", xtest_new.shape)

            self.ml_pipeline.x_train = xtrain_new
            self.ml_pipeline.x_test = xtest_new

            fld_path = self.ml_pipeline.job_data['job_data_path']
            fld_path = os.path.join(fld_path, DATA_FLD_NAME)

            pca_model_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "PCA.joblib")
            dump(pca, pca_model_path)

    def write_to_csv_and_update_status(self):
        # TODO write to csv and update status

        x_train = pd.DataFrame(self.ml_pipeline.x_train)
        x_test = pd.DataFrame(self.ml_pipeline.x_test)
        ytrain = self.ml_pipeline.y_train
        ytest = self.ml_pipeline.y_test

        x_train['Activation Status'] = ytrain
        x_test['Activation Status'] = ytest

        fld_path = self.ml_pipeline.job_data['job_data_path']
        fld_path = os.path.join(fld_path, DATA_FLD_NAME)

        train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "train.csv")
        test_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "test.csv")

        x_train.to_csv(train_file_path, index=False)
        x_test.to_csv(test_file_path, index=False)

        # update status
        self.ml_pipeline.status = "feature_extraction"

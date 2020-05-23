import MLPipeline
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

DATA_FLD_NAME = "step3"
DATA_FILE_NAME_PRFX = "FS_"


class FeatureSelection:

    def __init__(self, ml_pipeline: MLPipeline):
        print("Inside FeatureSelection initialization")

        self.ml_pipeline = ml_pipeline

        if self.ml_pipeline.status == "preprocessing":  # resuming at step 3
            print(ml_pipeline.job_data)

            print("Inside FeatureSelection ", self.ml_pipeline.x_train.shape)

            if self.ml_pipeline.data is None or self.ml_pipeline.x_train is None or self.ml_pipeline.x_test is None:
                padel_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step1", "FG_Padel.csv")
                train_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step2", "PP_train.csv")
                test_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step2", "PP_test.csv")

                data, data_labels = self.read_data(padel_data_fp)
                self.ml_pipeline.data = data
                self.ml_pipeline.data_labels = data_labels

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

            self.perform_feature_selection()

    def perform_feature_selection(self):
        self.perform_boruta_fs()

        # TODO check if all above steps successful
        self.write_to_csv_and_update_status()

    def perform_boruta_fs(self):

        if self.ml_pipeline.config.fs_boruta_flg:
            xtrain = self.ml_pipeline.x_train
            xtest = self.ml_pipeline.x_test
            ytrain = self.ml_pipeline.y_train

            print("Inside BorutaFS, Before Shape Train: ", xtrain.shape)
            print("Inside BorutaFS, Before Shape Test: ", xtest.shape)
            rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            boruta_selector = BorutaPy(rfc, n_estimators='auto', random_state=50)
            boruta_selector.fit(xtrain.values, ytrain)
            xtrain_sel = boruta_selector.transform(xtrain.values)
            xtest_sel = boruta_selector.transform(xtest.values)
            #     print(boruta_selector.support_)
            sel_cols = xtrain.columns[boruta_selector.support_]

            print("Inside BorutaFS, IN FeatureSelector get_feature_names ", sel_cols)

            train = pd.DataFrame(xtrain_sel, columns=sel_cols)
            test = pd.DataFrame(xtest_sel, columns=sel_cols)

            self.ml_pipeline.x_train = train
            self.ml_pipeline.x_test = test

            print("Inside BorutaFS, After Shape Train: ", train.shape)
            print("Inside BorutaFS,  After Shape Test: ", test.shape)

    def write_to_csv_and_update_status(self):
        # TODO write to csv and update status

        x_train = self.ml_pipeline.x_train
        x_test = self.ml_pipeline.x_test
        ytrain = self.ml_pipeline.y_train
        ytest = self.ml_pipeline.y_test

        # TODO add these (activation statuses to the files too)
        # x_train['Activation Status'] = ytrain
        # x_test['Activation Status'] = ytest

        fld_path = self.ml_pipeline.job_data['job_data_path']
        fld_path = os.path.join(fld_path, DATA_FLD_NAME)

        train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "train.csv")
        test_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "test.csv")

        # TODO remove these (activation statuses to the files too)
        train_labels_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "train_labels.csv")
        test_labels_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "test_labels.csv")

        x_train.to_csv(train_file_path, index=False)
        x_test.to_csv(test_file_path, index=False)

        ytrain_df = pd.DataFrame(ytrain)
        ytest_df = pd.DataFrame(ytest)

        ytrain_df.to_csv(train_labels_file_path, index=False)
        ytest_df.to_csv(test_labels_file_path, index=False)

        # update status
        self.ml_pipeline.status = "feature_selection"

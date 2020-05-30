import os

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from joblib import dump

import MLPipeline
import AppConfig as app_config
import ml_pipeline.utils.Helper as helper

DATA_FLD_NAME = app_config.FE_FLD_NAME
DATA_FILE_NAME_PRFX = app_config.FE_FLD_PREFIX


class FeatureExtraction:

    def __init__(self, ml_pipeline: MLPipeline, is_train: bool):

        self.ml_pipeline = ml_pipeline
        self.jlogger = self.ml_pipeline.jlogger

        self.is_train = is_train

        self.jlogger.info(
            "Inside FeatureExtraction initialization with status {} and is_train as {}".format(self.ml_pipeline.status,
                                                                                               self.is_train))

        # call only when in training state - inorder to reuse code for post preprocessing when not in training state
        if self.is_train:
            step4 = os.path.join(self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME)
            os.makedirs(step4, exist_ok=True)

            if self.ml_pipeline.status == app_config.STEP3_STATUS:  # resuming at step 4
                self.apply_on_all_fg()

    def apply_on_all_fg(self):

        if self.ml_pipeline.config.fg_padelpy_flg:
            self.jlogger.info("Started feature extraction of preprocessed PaDEL features")
            job_fld_path = self.ml_pipeline.job_data['job_fld_path']
            pp_padel_fld_path = os.path.join(
                *[job_fld_path, app_config.TEMP_TTS_FLD_NAME, app_config.FG_PADEL_FLD_NAME])

            padel_xtrain_fp = os.path.join(pp_padel_fld_path, app_config.TEMP_XTRAIN_FNAME)
            padel_ytrain_fp = os.path.join(pp_padel_fld_path, app_config.TEMP_YTRAIN_FNAME)
            padel_xtest_fp = os.path.join(pp_padel_fld_path, app_config.TEMP_XTEST_FNAME)
            padel_ytest_fp = os.path.join(pp_padel_fld_path, app_config.TEMP_YTEST_FNAME)

            self.ml_pipeline.x_train = pd.read_csv(padel_xtrain_fp)
            self.ml_pipeline.y_train = pd.read_csv(padel_ytrain_fp)

            self.ml_pipeline.x_test = pd.read_csv(padel_xtest_fp)
            self.ml_pipeline.y_test = pd.read_csv(padel_ytest_fp)

            # folder path to save output of preprocessed padel features feature extraction data
            fs_padel_fld_path = os.path.join(*[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME,
                                               app_config.FG_PADEL_FLD_NAME])
            self.fg_fe_fld_path = fs_padel_fld_path
            os.makedirs(self.fg_fe_fld_path, exist_ok=True)

            self.perform_feature_extraction()

            if self.ml_pipeline.config.fg_mordered_flg:
                self.jlogger.info("Started feature extraction of preprocessed mordred features")

                job_fld_path = self.ml_pipeline.job_data['job_fld_path']
                pp_mordred_fld_path = os.path.join(
                    *[job_fld_path, app_config.TEMP_TTS_FLD_NAME, app_config.FG_MORDRED_FLD_NAME])
                mordred_xtrain_fp = os.path.join(pp_mordred_fld_path, app_config.TEMP_XTRAIN_FNAME)
                mordred_ytrain_fp = os.path.join(pp_mordred_fld_path, app_config.TEMP_YTRAIN_FNAME)
                mordred_xtest_fp = os.path.join(pp_mordred_fld_path, app_config.TEMP_XTEST_FNAME)
                mordred_ytest_fp = os.path.join(pp_mordred_fld_path, app_config.TEMP_YTEST_FNAME)

                self.ml_pipeline.x_train = pd.read_csv(mordred_xtrain_fp)
                self.ml_pipeline.y_train = pd.read_csv(mordred_ytrain_fp)

                self.ml_pipeline.x_test = pd.read_csv(mordred_xtest_fp)
                self.ml_pipeline.y_test = pd.read_csv(mordred_ytest_fp)

                # folder path to save output of preprocessed mordred features feature extraction data
                fs_mordred_fld_path = os.path.join(*[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME,
                                                     app_config.FG_MORDRED_FLD_NAME])

                self.fg_fe_fld_path = fs_mordred_fld_path
                os.makedirs(self.fg_fe_fld_path, exist_ok=True)

                self.perform_feature_extraction()

            if self.is_train:
                updated_status = app_config.STEP4_STATUS

                job_oth_config_fp = self.ml_pipeline.job_data['job_oth_config_path']
                helper.update_job_status(job_oth_config_fp, updated_status)

                self.ml_pipeline.status = updated_status

                self.jlogger.info("Feature extraction completed successfully")

    def perform_feature_extraction(self):
        self.perform_pca()

        if self.is_train:
            self.write_final_data_to_csv()

    def perform_pca(self):

        if self.ml_pipeline.config.fe_pca_flg:
            pca_energy = self.ml_pipeline.config.fe_pca_energy

            xtrain = self.ml_pipeline.x_train
            xtest = self.ml_pipeline.x_test

            self.jlogger.info("Inside PCA, Before Shape Train: {}".format(xtrain.shape))
            self.jlogger.info("Inside PCA, Before Shape Test: {}".format(xtest.shape))
            pca = PCA(pca_energy)
            pca.fit(xtrain)

            xtrain_new = pca.transform(xtrain)
            xtest_new = pca.transform(xtest)
            self.jlogger.info("Inside PCA, After Shape Train: {}".format(xtrain_new.shape))
            self.jlogger.info("Inside PCA, After Shape Test: {}".format(xtest_new.shape))

            self.ml_pipeline.x_train = xtrain_new
            self.ml_pipeline.x_test = xtest_new

            if self.is_train:
                fld_path = self.fg_fe_fld_path
                pca_model_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "PCA.joblib")
                dump(pca, pca_model_path)

    def write_final_data_to_csv(self):
        x_train = pd.DataFrame(self.ml_pipeline.x_train)
        x_test = pd.DataFrame(self.ml_pipeline.x_test)
        ytrain = self.ml_pipeline.y_train
        ytest = self.ml_pipeline.y_test


        fld_path = self.fg_fe_fld_path

        train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "train.csv")
        test_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "test.csv")

        train_labels_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "train_labels.csv")
        test_labels_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "test_labels.csv")

        x_train.to_csv(train_file_path, index=False)
        x_test.to_csv(test_file_path, index=False)

        ytrain_df = pd.DataFrame(ytrain)
        ytest_df = pd.DataFrame(ytest)

        ytrain_df.to_csv(train_labels_file_path, index=False)
        ytest_df.to_csv(test_labels_file_path, index=False)

        self.save_final_data_copy_to_temp()

    def save_final_data_copy_to_temp(self):
        x_train = pd.DataFrame(self.ml_pipeline.x_train)
        x_test = pd.DataFrame(self.ml_pipeline.x_test)
        ytrain = self.ml_pipeline.y_train
        ytest = self.ml_pipeline.y_test

        # save a copy to temp folder for further processing in pipeline
        fg_fld_name = self.fg_fe_fld_path

        job_fld_path = self.ml_pipeline.job_data['job_fld_path']
        fld_path = os.path.join(job_fld_path, app_config.TEMP_TTS_FLD_NAME, os.path.basename(fg_fld_name))

        if not os.path.exists(fld_path):
            os.makedirs(fld_path, exist_ok=True)

        train_file_path = os.path.join(fld_path, app_config.TEMP_XTRAIN_FNAME)
        test_file_path = os.path.join(fld_path, app_config.TEMP_XTEST_FNAME)

        train_labels_file_path = os.path.join(fld_path, app_config.TEMP_YTRAIN_FNAME)
        test_labels_file_path = os.path.join(fld_path, app_config.TEMP_YTEST_FNAME)

        x_train.to_csv(train_file_path, index=False)
        x_test.to_csv(test_file_path, index=False)

        ytrain_df = pd.DataFrame(ytrain)
        ytest_df = pd.DataFrame(ytest)

        ytrain_df.to_csv(train_labels_file_path, index=False)
        ytest_df.to_csv(test_labels_file_path, index=False)

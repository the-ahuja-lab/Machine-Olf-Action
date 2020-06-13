import os
from collections import Counter

import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from joblib import dump

import MLPipeline
import AppConfig as app_config
import ml_pipeline.utils.Helper as helper

DATA_FLD_NAME = app_config.PP_FLD_NAME
DATA_FILE_NAME_PRFX = app_config.PP_FLD_PREFIX


class Preprocessing:

    def __init__(self, ml_pipeline: MLPipeline, is_train: bool):
        self.ml_pipeline = ml_pipeline
        self.jlogger = self.ml_pipeline.jlogger

        self.is_train = is_train

        self.jlogger.info(
            "Inside Preprocessing initialization with status {} and is_train as {}".format(self.ml_pipeline.status,
                                                                                           self.is_train))
        # call only when in training state - inorder to reuse code for post preprocessing when not in training state
        if self.is_train:
            step2 = os.path.join(self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME)
            os.makedirs(step2, exist_ok=True)

            if self.ml_pipeline.status == app_config.STEP1_STATUS:  # resuming at step 2
                self.apply_on_all_fg()

    def apply_on_all_fg(self):

        if self.ml_pipeline.config.fg_padelpy_flg:
            self.jlogger.info("Started pre-processing PaDEL features")
            padel_data_fp = os.path.join(*[self.ml_pipeline.job_data['job_data_path'], app_config.FG_FLD_NAME,
                                           app_config.FG_PADEL_FLD_NAME, app_config.FG_PADEL_FNAME])
            pp_padel_fld_path = os.path.join(*[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME,
                                               app_config.FG_PADEL_FLD_NAME])
            data, data_labels = self.read_data(padel_data_fp)
            self.ml_pipeline.data = data
            self.ml_pipeline.data_labels = data_labels

            # folder path to save output of padel features preprocessed data
            self.fg_pp_fld_path = pp_padel_fld_path
            os.makedirs(self.fg_pp_fld_path, exist_ok=True)

            pp_init_data_fpath = os.path.join(self.fg_pp_fld_path, DATA_FILE_NAME_PRFX + "init_data.csv")
            pp_init_labels_fpath = os.path.join(self.fg_pp_fld_path, DATA_FILE_NAME_PRFX + "init_labels.csv")

            data.to_csv(pp_init_data_fpath, index=False)
            data_labels.to_csv(pp_init_labels_fpath, index=False)

            self.preprocess_data()

        if self.ml_pipeline.config.fg_mordered_flg:
            self.jlogger.info("Started pre-processing mordred features")
            mordred_data_fp = os.path.join(*[self.ml_pipeline.job_data['job_data_path'], app_config.FG_FLD_NAME,
                                             app_config.FG_MORDRED_FLD_NAME, app_config.FG_MORDRED_FNAME])

            pp_mordred_fld_path = os.path.join(*[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME,
                                                 app_config.FG_MORDRED_FLD_NAME])
            data, data_labels = self.read_data(mordred_data_fp)
            self.ml_pipeline.data = data
            self.ml_pipeline.data_labels = data_labels

            # folder path to save output of mordred features preprocessed data
            self.fg_pp_fld_path = pp_mordred_fld_path
            os.makedirs(self.fg_pp_fld_path, exist_ok=True)

            pp_init_data_fpath = os.path.join(self.fg_pp_fld_path, DATA_FILE_NAME_PRFX + "init_data.csv")
            pp_init_labels_fpath = os.path.join(self.fg_pp_fld_path, DATA_FILE_NAME_PRFX + "init_labels.csv")

            data.to_csv(pp_init_data_fpath, index=False)
            data_labels.to_csv(pp_init_labels_fpath, index=False)

            self.preprocess_data()

        if self.is_train:
            updated_status = app_config.STEP2_STATUS

            job_oth_config_fp = self.ml_pipeline.job_data['job_oth_config_path']
            helper.update_job_status(job_oth_config_fp, updated_status)

            self.ml_pipeline.status = updated_status

        self.jlogger.info("Pre-processing completed successfully")

    def read_data(self, fp):
        data = pd.read_csv(fp)
        data_labels = data["Activation Status"]
        data = data.drop("Activation Status", axis=1)
        data = data.drop("CNAME", axis=1)
        try:
            data = data.drop("SMILES", axis=1)
        except:
            self.jlogger.warning("Don't have Smile's Column")
        data = self.coerce_df_columns_to_numeric(data, data.columns)
        return data, data_labels

    def coerce_df_columns_to_numeric(self, df, column_list):
        df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
        return df

    def preprocess_data(self):
        self.perform_column_pruning()
        self.perform_train_test_split()
        self.handle_missing_values()
        self.handle_class_imbalance()
        self.perform_data_normalization()
        self.handle_low_variance_cols()
        self.handle_correlated_cols()

        if self.is_train:
            self.write_final_data_to_csv()

    def perform_column_pruning(self):
        if self.ml_pipeline.config.pp_mv_col_pruning_flg and self.is_train:
            self.jlogger.info("Inside perform_column_pruning")
            th = self.ml_pipeline.config.pp_mv_col_pruning_th

            data = self.ml_pipeline.data

            data = data.replace(r'\s+', np.nan, regex=True)
            data[data == np.inf] = np.nan
            data = data.replace(r'^\s*$', np.nan, regex=True)

            na_sum_series = data.isna().sum()
            org_data = data.copy()

            NAN_data = pd.DataFrame({0: na_sum_series.index, 1: na_sum_series.values})
            dropped = []
            for i in range(len(NAN_data)):
                if NAN_data.iloc[i][1] >= th:  # TODO check if sum or sum/length i.e. avg greater than threshold
                    dropped.append(NAN_data.iloc[i][0])

            data = data.drop(dropped, axis=1)

            self.ml_pipeline.data = data

            # save output of the step if ml pipeline is in training state
            if self.is_train:
                fld_path = self.fg_pp_fld_path
                cols_pruned_fpath = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "cols_pruned.csv")
                data.to_csv(cols_pruned_fpath, index=False)

                NAN_fld_path = self.fg_pp_fld_path
                NAN_file_path = os.path.join(NAN_fld_path, DATA_FILE_NAME_PRFX + "FeatureWise_NANs.csv")
                org_data.isna().sum().to_csv(NAN_file_path,
                                             header=False)

                # save also the dropped columns list
                dropped_cols_lst_path = os.path.join(NAN_fld_path, DATA_FILE_NAME_PRFX + "Dropped_Features.txt")
                with open(dropped_cols_lst_path, 'w') as f:
                    for item in dropped:
                        f.write("%s\n" % item)

            self.jlogger.info("Dropped columns: {}".format(len(dropped)))
            self.jlogger.info("Data shape after pruning NAN values: {}".format(data.shape))

    def perform_train_test_split(self):
        # perform train test split only when in training state
        if self.is_train:
            self.jlogger.info("Inside perform_train_test_split")
            test_per = self.ml_pipeline.config.tts_test_per / 100
            data = self.ml_pipeline.data
            data_labels = self.ml_pipeline.data_labels

            x_train, x_test, y_train, y_test = train_test_split(data, data_labels.values, test_size=test_per,
                                                                random_state=100, stratify=data_labels.values)

            self.ml_pipeline.x_train = x_train
            self.ml_pipeline.x_test = x_test
            self.ml_pipeline.y_train = y_train
            self.ml_pipeline.y_test = y_test

            # save output of train test split
            fld_path = self.fg_pp_fld_path
            train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "init_train.csv")
            test_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "init_test.csv")

            x_train.to_csv(train_file_path, index=False)
            x_test.to_csv(test_file_path, index=False)

    def handle_missing_values(self):

        if self.ml_pipeline.config.pp_mv_imputation_flg:
            self.jlogger.info("Inside handle_missing_values")
            impute_mthd = self.ml_pipeline.config.pp_mv_imputation_mthd

            X_train = self.ml_pipeline.x_train
            X_test = self.ml_pipeline.x_test

            X_train = X_train.replace([np.inf, -np.inf, "", " "], np.nan)
            X_train = X_train.replace(["", " "], np.nan)
            X_test = X_test.replace([np.inf, -np.inf, "", " "], np.nan)
            X_test = X_test.replace(["", " "], np.nan)

            if impute_mthd == 'mean':
                X_train.fillna(X_train.mean(), inplace=True)
                X_test.fillna(X_train.mean(), inplace=True)
            else:
                self.jlogger.error("Mean imputing is only supported")
                raise ValueError("{} imputation method specified, only mean imputation supported".format(impute_mthd))

            self.ml_pipeline.x_train = X_train
            self.ml_pipeline.x_test = X_test

            # save output of the step if ml pipeline is in training state
            if self.is_train:
                fld_path = self.fg_pp_fld_path

                train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "mv_train.csv")
                test_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "mv_test.csv")

                X_train.to_csv(train_file_path, index=False)
                X_test.to_csv(test_file_path, index=False)

            self.jlogger.info(
                "Inside preprocessing, after handling missing values, train shape {}".format(X_train.shape))
            self.jlogger.info("Inside preprocessing, after handling missing values, test shape {}".format(X_test.shape))

    def handle_class_imbalance(self):

        if self.ml_pipeline.config.pp_climb_smote_flg:
            self.jlogger.info("Inside handle_class_imbalance using SMOTE")
            xtrain = self.ml_pipeline.x_train
            ytrain = self.ml_pipeline.y_train

            self.jlogger.info('Inside SMOTE, Original dataset shape %s' % Counter(ytrain))
            cols = xtrain.columns
            sm = SMOTE(random_state=50)
            xtrain_new, ytrain_new = sm.fit_resample(xtrain, ytrain)

            xtrain_df = pd.DataFrame(xtrain_new, columns=cols)
            self.jlogger.info('Inside SMOTE, Resampled dataset shape %s' % Counter(ytrain_new))

            self.ml_pipeline.x_train = xtrain_df
            self.ml_pipeline.y_train = ytrain_new

            # save output of the step if ml pipeline is in training state
            if self.is_train:
                fld_path = self.fg_pp_fld_path

                train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "smote_train.csv")

                xtrain_df.to_csv(train_file_path, index=False)

    def perform_data_normalization(self):

        if self.ml_pipeline.config.pp_normalization_flg:
            self.jlogger.info("Inside perform_data_normalization")
            traindata = self.ml_pipeline.x_train
            testdata = self.ml_pipeline.x_test

            try:
                x = traindata.values  # returns a numpy array
            except:
                x = traindata

            if self.ml_pipeline.config.pp_normalization_mthd == 'min_max':
                self.jlogger.info("Inside perform_data_normalization using minmax")
                min_max_scaler = preprocessing.MinMaxScaler()
                min_max_scaler.fit(x)

                x_scaled = min_max_scaler.transform(x)
                test = testdata.values
                test = min_max_scaler.transform(test)
                train_normal = pd.DataFrame(x_scaled)
                test_normal = pd.DataFrame(test)
                train_normal.columns = list(traindata.columns.values)
                test_normal.columns = list(testdata.columns.values)

                self.ml_pipeline.x_train = train_normal
                self.ml_pipeline.x_test = test_normal

                # save output of the step if ml pipeline is in training state
                if self.is_train:
                    fld_path = self.fg_pp_fld_path

                    train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "normalize_train.csv")
                    test_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "normalize_test.csv")

                    train_normal.to_csv(train_file_path, index=False)
                    test_normal.to_csv(test_file_path, index=False)

                    mm_scaler_model_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "data_normalization.joblib")
                    dump(min_max_scaler, mm_scaler_model_path)

        else:
            self.jlogger.error("Min-max normalization is only supported")
            raise ValueError("{} normalization method specified, only min-max normalization supported".format(
                self.ml_pipeline.config.pp_normalization_mthd))

    def handle_low_variance_cols(self):

        if self.ml_pipeline.config.pp_vt_flg:
            self.jlogger.info("Inside handle_low_variance_cols")
            th = self.ml_pipeline.config.pp_vt_th

            data_normal = self.ml_pipeline.x_train
            test_normal = self.ml_pipeline.x_test

            selector = VarianceThreshold(th)
            selector.fit(data_normal)
            data_var_free = data_normal[data_normal.columns[selector.get_support(indices=True)]]
            test_var_free = test_normal[test_normal.columns[selector.get_support(indices=True)]]

            self.ml_pipeline.x_train = data_var_free
            self.ml_pipeline.x_test = test_var_free

            # save output of the step if ml pipeline is in training state
            if self.is_train:
                fld_path = self.fg_pp_fld_path

                train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "varaince_train.csv")
                test_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "variance_test.csv")

                data_var_free.to_csv(train_file_path, index=False)
                test_var_free.to_csv(test_file_path, index=False)

            self.jlogger.info(
                "Inside preprocessing, after variance removal, train shape {}".format(data_var_free.shape))
            self.jlogger.info("Inside preprocessing, after variance removal, test shape {}".format(test_var_free.shape))

    def handle_correlated_cols(self):
        if self.ml_pipeline.config.pp_cr_flg:
            self.jlogger.info("Inside handle_correlated_cols")
            th = self.ml_pipeline.config.pp_cr_th

            traindata = self.ml_pipeline.x_train
            testdata = self.ml_pipeline.x_test

            corr_matrix = traindata.corr()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            to_drop = [column for column in upper.columns if any(upper[column] > th)]
            trainset = traindata.drop(traindata[to_drop], axis=1)
            testset = testdata.drop(testdata[to_drop], axis=1)

            self.ml_pipeline.x_train = trainset
            self.ml_pipeline.x_test = testset

            if self.is_train:
                fld_path = self.fg_pp_fld_path

                train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "corr_train.csv")
                test_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "corr_test.csv")

                trainset.to_csv(train_file_path, index=False)
                testset.to_csv(test_file_path, index=False)

            self.jlogger.info("Inside preprocessing, after correlation check, train shape {}".format(trainset.shape))
            self.jlogger.info("Inside preprocessing, after correlation check, test shape {}".format(testset.shape))

    def write_final_data_to_csv(self):
        x_train = self.ml_pipeline.x_train
        x_test = self.ml_pipeline.x_test
        ytrain = self.ml_pipeline.y_train
        ytest = self.ml_pipeline.y_test

        fld_path = self.fg_pp_fld_path

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
        x_train = self.ml_pipeline.x_train
        x_test = self.ml_pipeline.x_test
        ytrain = self.ml_pipeline.y_train
        ytest = self.ml_pipeline.y_test

        # save a copy to temp folder for further processing in pipeline
        fg_fld_name = self.fg_pp_fld_path

        job_fld_path = self.ml_pipeline.job_data['job_fld_path']
        fld_path = os.path.join(*[job_fld_path, app_config.TEMP_TTS_FLD_NAME, os.path.basename(fg_fld_name)])

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

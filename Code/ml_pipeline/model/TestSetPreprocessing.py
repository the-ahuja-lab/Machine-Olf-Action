import os
import pandas as pd
import numpy as np

from joblib import load

import MLPipeline
import AppConfig as app_config
import ml_pipeline.utils.Helper as helper


DATA_FLD_NAME = app_config.TSG_FLD_NAME


# TEST_FLD_NAME = app_config.TSG_TEST_FLD_NAME
#
# PADEL_FLD_RAW_NAME = app_config.TSG_RAW_FLD_NAME
# PADEL_FLD_PP_NAME = app_config.TSG_PP_FLD_NAME
# PADEL_FLD_PP_LIME_NAME = "pp_lime"

# TEST_CMPNDS_FLD_NAME = app_config.TSG_CMPND_FLD_NAME
#
# PP_FLD = app_config.PP_FLD_NAME
# PP_FIN_NAME = app_config.PP_FIN_XTRAIN_FNAME
# PP_NORM_NAME = app_config.PP_NORM_DUMP_NAME
#
# BORUTA_FLD = app_config.FS_FLD_NAME
# BORUTA_FS_NAME = app_config.FS_XTRAIN_FNAME
#
# PCA_FLD = app_config.FE_FLD_NAME
# PCA_MODEL = app_config.FE_PCA_DUMP_FNAME


class TestSetPreprocessing:

    def __init__(self, ml_pipeline: MLPipeline):

        self.ml_pipeline = ml_pipeline
        self.jlogger = self.ml_pipeline.jlogger

        self.jlogger.info("Inside TestSetPreprocessing initialization")

        if self.ml_pipeline.status == app_config.STEP6_STATUS:  # resuming at step 1
            self.apply_on_all_fg()

    def apply_on_all_fg(self):
        # Padel
        if self.ml_pipeline.config.fg_padelpy_flg:
            self.fg_fld_name = app_config.FG_PADEL_FLD_NAME
            self.preprocess_test_set()

        if self.ml_pipeline.config.fg_mordered_flg:
            # Mordred
            self.fg_fld_name = app_config.FG_MORDRED_FLD_NAME
            self.preprocess_test_set()

        updated_status = app_config.STEP6_1_STATUS

        job_oth_config_fp = self.ml_pipeline.job_data['job_oth_config_path']
        helper.update_job_status(job_oth_config_fp, updated_status)

        self.ml_pipeline.status = updated_status

        self.jlogger.info("Generated test set preprocessing completed successfully")

    def preprocess_test_set(self):
        padel_raw_fld_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, self.fg_fld_name,
              app_config.TSG_RAW_FLD_NAME])

        padel_pp_fld_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, self.fg_fld_name,
              app_config.TSG_PP_FLD_NAME])
        os.makedirs(padel_pp_fld_path, exist_ok=True)

        padel_pp_lime_fld_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, self.fg_fld_name,
              app_config.TSG_PP_LIME_FLD_NAME])
        os.makedirs(padel_pp_lime_fld_path, exist_ok=True)

        padel_test_cmpnd_fld_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, self.fg_fld_name,
              app_config.TSG_CMPND_FLD_NAME])
        os.makedirs(padel_test_cmpnd_fld_path, exist_ok=True)

        for file in os.listdir(padel_raw_fld_path):
            if file.endswith(".csv"):  # checking only csv files
                self.jlogger.info("Starting preprocessing {}".format(file))

                padel_fp = os.path.join(padel_raw_fld_path, file)
                cnames, padel_pp_lime_df, padel_pp_fin_df = self.preprocess_generated_test_set(padel_fp)

                cnames_df = pd.DataFrame(cnames, columns=["CNAME"])
                test_cmpnd_fp = os.path.join(padel_test_cmpnd_fld_path, file)
                cnames_df.to_csv(test_cmpnd_fp, index=False)

                padel_pp_fp = os.path.join(padel_pp_fld_path, file)
                padel_pp_fin_df.to_csv(padel_pp_fp, index=False)

                padel_pp_lime_fp = os.path.join(padel_pp_lime_fld_path, file)
                padel_pp_lime_df.to_csv(padel_pp_lime_fp, index=False)

    def preprocess_generated_test_set(self, padel_fp):
        df_test = pd.read_csv(padel_fp)
        compound_names = df_test['CNAME']

        self.jlogger.info("Before shape test {}".format(df_test.shape))

        df_init_train, init_features = self.extract_initial_train_features()
        df_init_test_fltrd = df_test[init_features]
        df_test_pp = self.apply_other_preprocess(df_init_train, df_init_test_fltrd)

        self.jlogger.info("After preprocessing shape test {}".format(df_test_pp.shape))

        df_fin_train, fin_features = self.extract_final_train_features()
        df_test_pp_final = df_test_pp[fin_features]

        self.jlogger.info("After feature selection shape test {}".format(df_test_pp_final.shape))

        test_final_np = self.apply_pca(df_test_pp_final)

        self.jlogger.info("After feature extraction shape test {}".format(test_final_np.shape))

        df_test_final = pd.DataFrame(test_final_np)

        return compound_names, df_test_pp_final, df_test_final

    def extract_initial_train_features(self):
        if self.ml_pipeline.config.pp_mv_col_pruning_flg:
            pp_train_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], app_config.PP_FLD_NAME, self.fg_fld_name,
                  app_config.PP_INIT_COL_PRUNED_FNAME])
        else:
            pp_train_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], app_config.PP_FLD_NAME, self.fg_fld_name,
                  app_config.PP_INIT_DATA_FNAME])

        df = pd.read_csv(pp_train_path)
        features = df.columns.to_list()

        return df, features

    def extract_final_train_features(self):
        fin_features = []
        df = None
        if self.ml_pipeline.config.fs_boruta_flg:
            boruta_train_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], app_config.FS_FLD_NAME, self.fg_fld_name,
                  app_config.FS_XTRAIN_FNAME])
            df = pd.read_csv(boruta_train_path)
            fin_features = df.columns.to_list()
        else:
            pp_train_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], app_config.PP_FLD_NAME, self.fg_fld_name,
                  app_config.PP_FIN_XTRAIN_FNAME])
            df = pd.read_csv(pp_train_path)
            fin_features = df.columns.to_list()

        return df, fin_features

    def handle_missing_values(self, df_train, df_test_fltrd):
        if self.ml_pipeline.config.pp_mv_imputation_flg:
            impute_mthd = self.ml_pipeline.config.pp_mv_imputation_mthd

            X_train = df_train
            X_test = df_test_fltrd

            X_test = X_test.replace(r'\s+', np.nan, regex=True)
            X_test = X_test.replace(r'^\s*$', np.nan, regex=True)
            X_test[X_test == np.inf] = np.nan

            X_test = X_test.replace([np.inf, -np.inf, "", " "], np.nan)
            X_test = X_test.replace(["", " "], np.nan)

            if impute_mthd == 'mean':
                X_test.fillna(X_train.mean(), inplace=True)
                df_test_fltrd = X_test.copy()
            else:
                self.jlogger.error("Mean imputing is only supported")
                raise ValueError("{} imputation method specified, only mean imputation supported".format(impute_mthd))

        return df_test_fltrd

    def perform_data_normalization(self, df_test_fltrd):

        if self.ml_pipeline.config.pp_normalization_flg:

            testdata = df_test_fltrd

            if self.ml_pipeline.config.pp_normalization_mthd == 'min_max':
                self.jlogger.info("Inside performing minmax normalization")

                pp_norm_model_path = os.path.join(
                    *[self.ml_pipeline.job_data['job_data_path'], app_config.PP_FLD_NAME, self.fg_fld_name,
                      app_config.PP_NORM_DUMP_NAME])

                min_max_scaler = load(pp_norm_model_path)

                # min_max_scaler = preprocessing.MinMaxScaler()

                test = testdata.values
                test = min_max_scaler.transform(test)
                test_normal = pd.DataFrame(test)
                test_normal.columns = list(testdata.columns.values)

                df_test_fltrd = test_normal
            else:
                self.jlogger.error("Min-max normalization is only supported")
                raise ValueError("{} normalization method specified, only min-max normalization supported".format(
                    self.ml_pipeline.config.pp_normalization_mthd))

        return df_test_fltrd

    def apply_pca(self, df_test_fltrd):
        if self.ml_pipeline.config.fe_pca_flg:
            xtest = df_test_fltrd.copy()

            pca_model_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], app_config.FE_FLD_NAME, self.fg_fld_name,
                  app_config.FE_PCA_DUMP_FNAME])

            pca = load(pca_model_path)

            self.jlogger.info("Inside PCA, Before Shape Test: {}".format(xtest.shape))
            xtest_new = pca.transform(xtest)
            self.jlogger.info("Inside PCA, After Shape Test: {}".format(xtest_new.shape))

            pca_op = xtest_new

        else:
            pca_op = df_test_fltrd.values

        return pca_op

    def apply_other_preprocess(self, df_train, df_test_fltrd):
        df_test_fltrd = self.handle_missing_values(df_train, df_test_fltrd)
        df_test_fltrd = self.perform_data_normalization(df_test_fltrd)
        return df_test_fltrd

import os
import pandas as pd

import pickle

import MLPipeline
import AppConfig as app_config
import ml_pipeline.utils.Helper as helper

import LIMEExplanation


DATA_FLD_NAME = app_config.TSG_FLD_NAME

ALL_TEST_DF = {}
ALL_TEST_COMPOUNDS = {}


class TestSetPrediction:

    def __init__(self, ml_pipeline: MLPipeline):
        self.ml_pipeline = ml_pipeline
        self.jlogger = self.ml_pipeline.jlogger

        self.jlogger.info("Inside TestSetPrediction initialization")

        self.lime_exp = None

        if self.ml_pipeline.status == app_config.STEP6_1_STATUS:  # resuming at step 6
            self.apply_on_all_fg()

    def apply_on_all_fg(self):
        # Padel
        if self.ml_pipeline.config.fg_padelpy_flg:
            self.fg_fld_name = app_config.FG_PADEL_FLD_NAME
            self.initialize_lime_explanation()
            self.apply_classification_models()

        if self.ml_pipeline.config.fg_mordered_flg:
            # Mordred
            self.fg_fld_name = app_config.FG_MORDRED_FLD_NAME
            self.initialize_lime_explanation()
            self.apply_classification_models()

        updated_status = app_config.STEPS_COMPLETED_STATUS

        job_oth_config_fp = self.ml_pipeline.job_data['job_oth_config_path']
        helper.update_job_status(job_oth_config_fp, updated_status)

        self.ml_pipeline.status = updated_status

        self.jlogger.info("Generated test set prediction completed successfully")

    def initialize_lime_explanation(self):
        if self.ml_pipeline.config.exp_lime_flg:
            lime_ml_pipeline = MLPipeline.MLPipeline(self.ml_pipeline.job_id)
            # lime_ml_pipeline.status = "test_set_generation"
            self.lime_exp = LIMEExplanation.LIMEExplanation(lime_ml_pipeline)
            self.lime_exp.fg_fld_name = self.fg_fld_name
            self.lime_exp.fetch_train_data()

    def apply_classification_models(self):
        self.apply_gbm()
        self.apply_svm()
        self.apply_rf()
        self.apply_lr()
        self.apply_gnb()
        self.apply_et()
        self.apply_mlp()

    def load_all_test_files(self):
        if len(ALL_TEST_DF) == 0:

            padel_pp_fld_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, self.fg_fld_name,
                  app_config.TSG_PP_FLD_NAME])

            test_cmpnd_fld_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, self.fg_fld_name,
                  app_config.TSG_CMPND_FLD_NAME])

            # all_test_compounds_df = pd.read_csv(test_cmpnd_fld_path)
            # ALL_TEST_COMPOUNDS = all_test_compounds_df[all_test_compounds_df.columns[0]]

            for file in os.listdir(padel_pp_fld_path):
                self.jlogger.debug("Collecting test preprocessed file {}".format(file))

                if file.endswith(".csv"):  # checking only csv files
                    padel_test_fp = os.path.join(padel_pp_fld_path, file)
                    df = pd.read_csv(padel_test_fp)
                    ALL_TEST_DF[file] = df

            for file in os.listdir(test_cmpnd_fld_path):
                self.jlogger.debug("Collecting test compound names from {}".format(file))

                if file.endswith(".csv"):  # checking only csv files
                    cmpnd_name_fp = os.path.join(test_cmpnd_fld_path, file)
                    df = pd.read_csv(cmpnd_name_fp)
                    ALL_TEST_COMPOUNDS[file] = df[df.columns[0]]

        return ALL_TEST_DF, ALL_TEST_COMPOUNDS

    def apply_model_for_predictions(self, model, df, all_test_compounds):

        padel_test_df = df.copy()

        pred_labels = model.predict(padel_test_df)
        try:
            prob = model.predict_proba(padel_test_df)
        except:
            print("This model does not support probability scores")
            prob = []
        padel_test_df['CNAME'] = all_test_compounds

        novel_compounds_predictions = pd.DataFrame(list(zip(padel_test_df['CNAME'], pred_labels, prob)),
                                                   columns=['Compound Name', 'Predicted Label',
                                                            'Predicted probability'])
        return novel_compounds_predictions

    def fetch_model_save_predictions(self, model_name):
        model_pkl_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], app_config.CLF_FLD_NAME, self.fg_fld_name, model_name,
              "clf_" + model_name + ".pkl"])

        with open(model_pkl_path, 'rb') as f:
            model = pickle.load(f)

        all_test_df, all_test_compounds = self.load_all_test_files()

        if not self.lime_exp is None:
            self.lime_exp.lime_explainer = None

        for padel_fname, test_df in all_test_df.items():
            test_compounds = all_test_compounds[padel_fname]
            novel_compounds_predictions = self.apply_model_for_predictions(model, test_df, test_compounds)

            novel_pred_fld_p = os.path.join(
                *[self.ml_pipeline.job_data['job_results_path'], self.fg_fld_name, app_config.NOVEL_RESULTS_FLD_NAME,
                  model_name])
            os.makedirs(novel_pred_fld_p, exist_ok=True)

            # TODO Add model name in prediction file
            pred_f_name = "pred_" + model_name + "_" + padel_fname
            novel_pred_fp = os.path.join(novel_pred_fld_p, pred_f_name)

            novel_compounds_predictions.to_csv(novel_pred_fp, index=False)

            if self.ml_pipeline.config.exp_lime_flg:
                lime_exp_f_name = "lime_exp_" + model_name + "_" + helper.change_ext(padel_fname, ".csv", ".pdf")
                lime_exp_pdf_fp = os.path.join(novel_pred_fld_p, lime_exp_f_name)

                self.lime_exp.exp_preds_using_lime(model, test_compounds, padel_fname, lime_exp_pdf_fp)


    def apply_gnb(self):
        if self.ml_pipeline.config.clf_gnb_flg:
            self.fetch_model_save_predictions("gnb")

            if self.ml_pipeline.config.clf_bagging_gnb:
                self.fetch_model_save_predictions("gnb_bagging")

    def apply_gbm(self):
        if self.ml_pipeline.config.clf_gbm_flg:
            self.fetch_model_save_predictions("gbm")

            if self.ml_pipeline.config.clf_bagging_gbm:
                self.fetch_model_save_predictions("gbm_bagging")

    def apply_svm(self):
        if self.ml_pipeline.config.clf_svm_flg:
            self.fetch_model_save_predictions("svm")

            if self.ml_pipeline.config.clf_bagging_svm:
                self.fetch_model_save_predictions("svm_bagging")

    def apply_lr(self):
        if self.ml_pipeline.config.clf_lr_flg:
            self.fetch_model_save_predictions("lr")

            if self.ml_pipeline.config.clf_bagging_lr:
                self.fetch_model_save_predictions("lr_bagging")

    def apply_rf(self):
        if self.ml_pipeline.config.clf_rf_flg:
            self.fetch_model_save_predictions("rf")

            if self.ml_pipeline.config.clf_bagging_rf:
                self.fetch_model_save_predictions("rf_bagging")

    def apply_et(self):
        if self.ml_pipeline.config.clf_et_flg:
            self.fetch_model_save_predictions("et")

            if self.ml_pipeline.config.clf_bagging_et:
                self.fetch_model_save_predictions("et_bagging")

    def apply_mlp(self):
        if self.ml_pipeline.config.clf_mlp_flg:
            self.fetch_model_save_predictions("mlp")

            if self.ml_pipeline.config.clf_bagging_mlp:
                self.fetch_model_save_predictions("mlp_bagging")

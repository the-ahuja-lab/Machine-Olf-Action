import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from joblib import dump, load

import MLPipeline
from CompoundSimilarity import CompoundSimilarity
from ml_pipeline.settings import APP_STATIC

import pickle

import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from sklearn.base import clone

DATA_FLD_NAME = "step6"
TEST_FLD_NAME = "test"
PADEL_FLD_NAME = "fg_padel"
PADEL_FLD_PP_NAME = "preprocessed"
PADEL_FLD_PP_LIME_NAME = "pp_lime"

TEST_CMPNDS_FLD_NAME = "test_compounds"
TEST_CMPNDS_FILE_NAME = "test_compounds.csv"

RESULTS_FLD_NAME = "novel_predictions"

MODEL_FLD = "step5"

ALL_TEST_DF = {}
ALL_TEST_COMPOUNDS = {}

gnb_explainer = None


class LIMEExplanation:

    def __init__(self, ml_pipeline: MLPipeline):
        print("Inside TestSetPrediction initialization")

        self.ml_pipeline = ml_pipeline

        self.lime_explainer = None

        # TODO change status from test_set_generation to test_set_prediction
        if self.ml_pipeline.status == "test_set_generation":  # resuming at step 1
            self.fetch_train_data()

    def fetch_train_data(self):
        # TODO if boruta is enable then do otherwise fetch from previous step
        train_data_fp = os.path.join(self.ml_pipeline.job_data['job_data_path'], "step3", "FS_train.csv")
        test_data_fp = os.path.join(self.ml_pipeline.job_data['job_data_path'], "step3", "FS_test.csv")

        train_data_labels_fp = os.path.join(self.ml_pipeline.job_data['job_data_path'], "step3", "FS_train_labels.csv")
        test_data_labels_fp = os.path.join(self.ml_pipeline.job_data['job_data_path'], "step3", "FS_test_labels.csv")

        x_train = pd.read_csv(train_data_fp)
        y_train_df = pd.read_csv(train_data_labels_fp)
        y_train = y_train_df[y_train_df.columns[0]]

        x_test = pd.read_csv(test_data_fp)
        y_test_df = pd.read_csv(test_data_labels_fp)
        y_test = y_test_df[y_test_df.columns[0]]

        self.ml_pipeline.x_train = x_train
        self.ml_pipeline.x_test = x_test
        self.ml_pipeline.y_train = y_train
        self.ml_pipeline.y_test = y_test

    def get_explainer(self, model):
        if self.lime_explainer is None:
            xtrain = self.ml_pipeline.x_train
            ytrain = self.ml_pipeline.y_train

            self.lime_explainer = LimeTabularExplainer(xtrain.values, mode='classification', training_labels=ytrain,
                                                       feature_names=xtrain.columns)

            print(model.get_params())
            new_model = clone(model)
            new_model.fit(self.ml_pipeline.x_train, self.ml_pipeline.y_train)
            self.model = new_model

            print(self.model)

    def exp_preds_using_lime(self, model, test_compounds, test_features_fname, result_pdf_fp):
        self.figcount = 0
        self.Figureset = []

        self.get_explainer(model)

        padel_pp_lime_fld_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, PADEL_FLD_NAME, PADEL_FLD_PP_LIME_NAME])

        padel_test_fp = os.path.join(padel_pp_lime_fld_path, test_features_fname)
        xtest = pd.read_csv(padel_test_fp)

        print("Getting lime predictions for ", test_features_fname, " total data points ", len(xtest))

        explainer = self.lime_explainer
        for j in range(len(xtest)):
            print(j)
            # change labels to  [0, 1], if want to calculate for both the classes
            labels = [1]
            exp = explainer.explain_instance(xtest.iloc[j], self.model.predict_proba, num_features=25, labels=labels)
            self.plot_explanation(exp, labels, test_compounds[j])

        self.plot_all_figures(result_pdf_fp)

    def plot_explanation(self, exp_inst, labels, compound_name):

        fig = plt.figure(num=self.figcount, clear=True)

        for i, label in enumerate(labels):
            exp = exp_inst.as_list(label=label)
            plt.subplot(1, len(labels), i + 1)

            vals = [x[1] for x in exp]
            names = [x[0] for x in exp]
            vals.reverse()
            names.reverse()
            colors = ['green' if x > 0 else 'red' for x in vals]
            pos = np.arange(len(exp)) + .5
            plt.tight_layout()
            plt.barh(pos, vals, align='center', color=colors)
            plt.yticks(pos, names)
            if exp_inst.mode == "classification":
                title = compound_name + ' (Local explanation for class %s' % exp_inst.class_names[label] + ")"
            else:
                title = 'Local explanation'
            plt.title(title)

        self.figcount += 1
        self.Figureset.append(fig)

    def plot_all_figures(self, pdf_file_path):
        figlist = self.Figureset
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_file_path)
        for i in range(len(figlist)):
            pdf.savefig(figlist[i], bbox_inches="tight")
        pdf.close()

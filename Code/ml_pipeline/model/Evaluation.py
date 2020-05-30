from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import BaggingClassifier

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import seaborn as sns

# import rpy2
# import rpy2.robjects as robjects  # robjects as python objects
# import rpy2.robjects.packages as rpackages  # helps download and import r packages

import os
import pandas as pd
import numpy as np

from collections import OrderedDict

import pickle

import MLPipeline
import AppConfig as app_config

DATA_FLD_NAME = app_config.CLF_FLD_NAME
DATA_FILE_NAME_PRFX = app_config.CLF_FLD_PREFIX

# BAGGING_FLD_NAME = "bagging"

RESULTS_FLD_NAME = app_config.CLF_RESULTS_FLD_NAME


# figcount = 0
# Figureset = []


class Evaluation:

    def __init__(self, ml_pipeline: MLPipeline):
        self.ml_pipeline = ml_pipeline
        self.jlogger = self.ml_pipeline.jlogger
        self.figcount = 0
        self.Figureset = []

    def evaluate_and_save_results(self, model, fld_name):
        x_train = self.ml_pipeline.x_train
        y_train = self.ml_pipeline.y_train

        x_test = self.ml_pipeline.x_test
        y_test = self.ml_pipeline.y_test

        fg_fld_name = os.path.basename(self.ml_pipeline.fg_clf_fld_path)

        fld_path = self.ml_pipeline.job_data['job_data_path']
        fld_path = os.path.join(*[fld_path, DATA_FLD_NAME, fg_fld_name, fld_name])

        os.makedirs(fld_path, exist_ok=True)

        clf_pkl_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + fld_name + ".pkl")

        train_preds_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "train_preds.csv")
        test_preds_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "test_preds.csv")

        with open(clf_pkl_path, 'wb') as f:
            pickle.dump(model, f)

        y_train_preds = model.predict(x_train)
        y_train_probas = model.predict_proba(x_train)

        y_test_preds = model.predict(x_test)
        y_test_probas = model.predict_proba(x_test)

        df_train = pd.DataFrame([], columns=['Prediction', 'Ground Truth', 'Prob 0', 'Prob 1'])
        df_train['Prediction'] = y_train_preds
        df_train['Ground Truth'] = y_train
        df_train['Prob 0'] = y_train_probas[:, 0]
        df_train['Prob 1'] = y_train_probas[:, 1]
        df_train.to_csv(train_preds_path, index=False)

        df_test = pd.DataFrame([], columns=['Prediction', 'Ground Truth', 'Prob 0', 'Prob 1'])
        df_test['Prediction'] = y_test_preds
        df_test['Ground Truth'] = y_test
        df_test['Prob 0'] = y_test_probas[:, 0]
        df_test['Prob 1'] = y_test_probas[:, 1]
        df_test.to_csv(test_preds_path, index=False)

        self.save_results(fld_name, df_train, df_test)

    def save_results(self, fld_name, df_train, df_test):
        # global Figureset

        y_train = df_train['Ground Truth']
        y_train_preds = df_train['Prediction']

        train_prob0 = df_train['Prob 0'].to_numpy()
        train_prob1 = df_train['Prob 1'].to_numpy()

        train_np_yprobas = np.c_[train_prob0, train_prob1]

        y_test = df_test['Ground Truth']
        y_test_preds = df_test['Prediction']
        test_prob0 = df_test['Prob 0'].to_numpy()
        test_prob1 = df_test['Prob 1'].to_numpy()

        test_np_yprobas = np.c_[test_prob0, test_prob1]

        train_fld_name = fld_name + "_train"
        test_fld_name = fld_name + "_test"

        self.save_all_model_plots(y_train_preds, y_train, train_np_yprobas, train_fld_name)
        self.save_all_model_plots(y_test_preds, y_test, test_np_yprobas, test_fld_name)

        self.save_plots_pdf(fld_name)

    def save_plots_pdf(self, fld_name):

        fg_fld_name = os.path.basename(self.ml_pipeline.fg_clf_fld_path)

        fld_path = self.ml_pipeline.job_data['job_results_path']
        fld_path = os.path.join(*[fld_path, fg_fld_name, RESULTS_FLD_NAME])

        os.makedirs(fld_path, exist_ok=True)

        pdf_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + fld_name + ".pdf")
        self.plot_all_figures(pdf_file_path)

    def save_all_model_plots(self, np_ypreds, np_ytrues, np_yprobas, title):
        self.print_confusion_matrix(np_ytrues, np_ypreds, "Confusion Matrix - " + title)

        gt = np_ytrues.tolist()
        probs = np_yprobas[:, 1].tolist()
        fpr, tpr = self.get_smoothened_fpr_tpr_from_pROC(gt, probs)
        self.plot_r_smoothened_curve(fpr, tpr, "ROC Curve - " + title)

        self.print_roc_curve(np_ytrues, np_ypreds, np_yprobas, "ROC Curve - " + title)
        res = self.evaluate_model(np_ypreds, np_ytrues, np_yprobas)

        self.jlogger.info("Evaluation of {} {}".format(title, res))

    def evaluate_all_bagged_clf(self, bc, n, x_test, y_test):
        res_list = []
        iters = []
        res_dict_keys = {}

        for i in range(n):
            if i % 100 == 0:
                print("Completed Iter: " + str(i))

            clf = bc.estimators_[i]

            ypred = clf.predict(x_test)
            yproba = clf.predict_proba(x_test)

            res = self.evaluate_model(ypred, y_test, yproba)
            res_list.append(list(res.values()))

            res_dict_keys = list(res.keys())

            iters.append(i + 1)
        #         print("-----------------------")
        results = pd.DataFrame(res_list, columns=res_dict_keys)

        return results

    def evaluate_and_save_bagging_results(self, bc, n, x_test, y_test, title, fld_path):
        # # resetting all figures
        # self.figcount = 0
        # self.Figureset = []

        # evaluate aggregated bagging clf
        bc_ypred = bc.predict(x_test)
        bc_yproba = bc.predict_proba(x_test)

        df_test = pd.DataFrame([], columns=['Prediction', 'Ground Truth', 'Prob 0', 'Prob 1'])
        df_test['Prediction'] = bc_ypred
        df_test['Ground Truth'] = y_test
        df_test['Prob 0'] = bc_yproba[:, 0]
        df_test['Prob 1'] = bc_yproba[:, 1]

        test_preds_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + title + ".csv")
        df_test.to_csv(test_preds_path, index=False)

        self.save_all_model_plots(bc_ypred, y_test, bc_yproba, title)

        # evaluate each of the bagged classifier
        results = self.evaluate_all_bagged_clf(bc, n, x_test, y_test)
        self.plot_box_plot_results(results, title)

        # fg_fld_name = os.path.basename(self.ml_pipeline.fg_clf_fld_path)
        # res_fld_path = self.ml_pipeline.job_data['job_results_path']
        # res_fld_path = os.path.join(*[res_fld_path, fg_fld_name, RESULTS_FLD_NAME])
        #
        # pdf_file_path = os.path.join(res_fld_path, DATA_FILE_NAME_PRFX + title + ".pdf")
        # self.plot_all_figures(pdf_file_path)

        iter_results_csv_path = os.path.join(fld_path,
                                             DATA_FILE_NAME_PRFX + title + " Iteration wise Evaluation Results.csv")
        stats_results_csv_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + title + " Evaluation Stats.csv")

        results.to_csv(iter_results_csv_path, float_format='%.4f')
        results.describe().to_csv(stats_results_csv_path, float_format='%.4f')

    def evaluate_bagging_model(self, clf, n, fld_name):
        # resetting all figures
        self.figcount = 0
        self.Figureset = []

        # convert to int in case float
        n = int(n)

        fg_fld_name = os.path.basename(self.ml_pipeline.fg_clf_fld_path)
        fld_path = self.ml_pipeline.job_data['job_data_path']
        fld_path = os.path.join(*[fld_path, DATA_FLD_NAME, fg_fld_name, fld_name])

        os.makedirs(fld_path, exist_ok=True)

        x_train = self.ml_pipeline.x_train
        y_train = self.ml_pipeline.y_train
        x_test = self.ml_pipeline.x_test
        y_test = self.ml_pipeline.y_test

        bc = BaggingClassifier(base_estimator=clf, n_estimators=n, bootstrap=True, n_jobs=-1, random_state=42)
        bc.fit(x_train, y_train)

        clf_pkl_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + fld_name + ".pkl")
        with open(clf_pkl_path, 'wb') as f:
            pickle.dump(bc, f)

        self.evaluate_and_save_bagging_results(bc, n, x_train, y_train, "Training - " + fld_name, fld_path)
        self.evaluate_and_save_bagging_results(bc, n, x_test, y_test, "Testing - " + fld_name, fld_path)

        fg_fld_name = os.path.basename(self.ml_pipeline.fg_clf_fld_path)
        res_fld_path = self.ml_pipeline.job_data['job_results_path']
        res_fld_path = os.path.join(*[res_fld_path, fg_fld_name, RESULTS_FLD_NAME])

        pdf_file_path = os.path.join(res_fld_path, DATA_FILE_NAME_PRFX + fld_name + ".pdf")
        self.plot_all_figures(pdf_file_path)

    def evaluate_model(self, ytest_pred, ytest, ytest_probas):
        """
        This method evaluates a model on various metrics. The evaluation happens per sample basis and not on
        aggregated individual basis.
        """
        prf = precision_recall_fscore_support(ytest, ytest_pred, average="macro")

        accuracy = accuracy_score(ytest, ytest_pred)

        mcc = matthews_corrcoef(ytest, ytest_pred)

        tn, fp, fn, tp = confusion_matrix(ytest, ytest_pred).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)

        kappa = cohen_kappa_score(ytest, ytest_pred)

        fpr, tpr, thresholds = metrics.roc_curve(ytest, ytest_probas[:, 1])
        aucroc = metrics.auc(fpr, tpr)

        ap = average_precision_score(ytest, ytest_probas[:, 1])

        res = OrderedDict()
        res["Accuracy"] = accuracy
        res["Precision"] = prf[0]
        res["Recall"] = prf[1]
        res["F1-Score"] = prf[2]
        res["MCC"] = mcc
        res["Specificity"] = specificity
        res["Sensitivity"] = sensitivity
        res["Kappa"] = kappa
        res["AUCROC"] = aucroc
        res["AP"] = ap

        return res

    def print_confusion_matrix(self, testlabel, y_pred, title_name):
        # global figcount
        # global Figureset
        cf = confusion_matrix(testlabel, y_pred)
        df_cm = pd.DataFrame(cf, index=[0, 1], columns=[0, 1])

        fig = plt.figure(self.figcount, clear=True)

        plt.title(title_name)
        sns.heatmap(df_cm, annot=True, fmt='d')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')

        self.figcount += 1
        self.Figureset.append(fig)

        # plt.show()

    def get_smoothened_fpr_tpr_from_pROC(self, gt, probs):
        # print("Inside get_smoothened_fpr_tpr_from_pROC ", len(gt), len(probs))
        # try:
        #     package = "pROC"
        #     if not rpackages.isinstalled(package):
        #         # import R's utility package
        #         utils = rpackages.importr('utils')
        #         utils.chooseCRANmirror(ind=1)
        #         print("Installing PROC")
        #         utils.install_packages('pROC')
        #
        #         pROC = rpackages.importr('pROC')
        #
        #     gt_r = robjects.IntVector(gt)
        #     probs_r = robjects.FloatVector(probs)
        #
        #     # r function string to execute
        #     r_roc_plot_func = """
        # plot_smooth_roc <- function(gt, probs){
        # fin_roc <- roc(gt,probs,smooth=TRUE)
        # se = fin_roc[['sensitivities']]
        # sp = fin_roc[['specificities']]
        #
        # fpr <- 1 - sp
        # tpr <- se
        #
        # return(list(fpr = fpr, tpr = tpr))
        # }"""
        #
        #     robjects.r(r_roc_plot_func)  # adding r function to global environment to refer later
        #     r_plot_smooth_roc = robjects.r['plot_smooth_roc']  # r function from function string
        #
        #     res = r_plot_smooth_roc(gt_r, probs_r)  # calling defined r function
        #
        #     fpr = np.array(res.rx2('fpr'))  # storing the return value as numpy array, fpr and tpr
        #     tpr = np.array(res.rx2('tpr'))
        # except:
        #     print("Error calling R smoothened ROC function")
        #     fpr, tpr, _ = roc_curve(gt, probs)
        fpr, tpr, _ = roc_curve(gt, probs)
        return fpr, tpr

    def plot_r_smoothened_curve(self, fpr, tpr, title_name):
        # global figcount
        # global Figureset
        roc_auc = auc(fpr, tpr)
        # print("AUC VALUE :", roc_auc)

        fig = plt.figure(self.figcount, clear=True)

        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.title(title_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title_name)
        plt.legend(loc="lower right")

        self.figcount += 1
        self.Figureset.append(fig)

        # plt.show()

    def print_roc_curve(self, testlabel, y_pred, y_proba, title_name):
        # global figcount
        # global Figureset
        fpr, tpr, _ = roc_curve(testlabel, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        # print("AUC VALUE :", roc_auc)

        fig = plt.figure(self.figcount, clear=True)

        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.title(title_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title_name)
        plt.legend(loc="lower right")

        self.figcount += 1
        self.Figureset.append(fig)

        # plt.show()

    # TODO Box Plot for Bagging
    def plot_box_plot_results(self, results, title):
        # global figcount
        # global Figureset
        fig = plt.figure(num=self.figcount, figsize=(10, 5), clear=True)
        plt.boxplot(results.T)
        plt.xticks(np.arange(1, len(results.columns) + 1), results.columns, rotation=45)
        plt.title("Evaluation metrics distribution - " + title)
        self.figcount += 1
        self.Figureset.append(fig)

    def plot_all_figures(self, pdf_file_path):
        figlist = self.Figureset
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_file_path)
        for i in range(len(figlist)):
            pdf.savefig(figlist[i], bbox_inches="tight")
        pdf.close()

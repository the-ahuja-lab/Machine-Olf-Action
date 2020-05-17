import MLPipeline
import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

import pickle
import Evaluation

DATA_FLD_NAME = "step5"
DATA_FILE_NAME_PRFX = "clf_"


class Classification:

    def __init__(self, ml_pipeline: MLPipeline):
        print("Inside Classification initialization")

        self.ml_pipeline = ml_pipeline

        if self.ml_pipeline.status == "feature_extraction":  # resuming at step 5
            print(ml_pipeline.job_data)

            if self.ml_pipeline.data is None or self.ml_pipeline.x_train is None or self.ml_pipeline.x_test is None:
                padel_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step1", "FG_Padel.csv")
                train_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step4", "FE_train.csv")
                test_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step4", "FE_test.csv")

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

            self.apply_classification_models()

    def apply_classification_models(self):
        self.apply_gbm()
        self.apply_svm()
        self.apply_rf()
        self.apply_lr()
        self.apply_gnb()
        self.apply_et()
        self.apply_mlp()

    def apply_gbm(self):

        if self.ml_pipeline.config.clf_gbm_flg:

            if self.ml_pipeline.config.clf_gbm_auto:
                x_train = self.ml_pipeline.x_train
                y_train = self.ml_pipeline.y_train

                #TODO perform grid search here
                clf = GradientBoostingClassifier(n_estimators=50, random_state=None, max_depth=2)
                chosen_model = clf.fit(x_train, y_train)
            else:
                manual_params = self.ml_pipeline.config.clf_gbm_manual

            evalclf = Evaluation.Evaluation(self.ml_pipeline)
            evalclf.evaluate_and_save_results(chosen_model, "gbm")

            if self.ml_pipeline.config.clf_bagging_gbm:
                n = self.ml_pipeline.config.clf_bag_gbm_n
                evalclf.evaluate_bagging_model(chosen_model, n, "gbm_bagging")

    def apply_et(self):

        if self.ml_pipeline.config.clf_et_flg:

            if self.ml_pipeline.config.clf_et_auto:
                x_train = self.ml_pipeline.x_train
                y_train = self.ml_pipeline.y_train

                # TODO perform grid search here
                clf = ExtraTreesClassifier(n_estimators=200, random_state=42, max_depth=10, n_jobs=-1)
                chosen_model = clf.fit(x_train, y_train)
            else:
                manual_params = self.ml_pipeline.config.clf_gbm_manual

            evalclf = Evaluation.Evaluation(self.ml_pipeline)
            evalclf.evaluate_and_save_results(chosen_model, "et")

            if self.ml_pipeline.config.clf_bagging_et:
                n = self.ml_pipeline.config.clf_bag_et_n
                evalclf.evaluate_bagging_model(chosen_model, n, "et_bagging")

    def apply_svm(self):

        if self.ml_pipeline.config.clf_svm_flg:

            x_train = self.ml_pipeline.x_train
            y_train = self.ml_pipeline.y_train

            if self.ml_pipeline.config.clf_svm_auto:
                grid_search_model = self.SVM_GridSearch()
                grid_search_model.fit(x_train, y_train)
                chosen_model = grid_search_model.best_estimator_
                print(chosen_model)
            else:
                manual_params = self.ml_pipeline.config.clf_svm_manual

            evalclf = Evaluation.Evaluation(self.ml_pipeline)
            evalclf.evaluate_and_save_results(chosen_model, "svm")

            if self.ml_pipeline.config.clf_bagging_svm:
                n = self.ml_pipeline.config.clf_bag_svm_n
                evalclf.evaluate_bagging_model(chosen_model, n, "svm_bagging")


    def apply_rf(self):

        if self.ml_pipeline.config.clf_rf_flg:

            x_train = self.ml_pipeline.x_train
            y_train = self.ml_pipeline.y_train

            if self.ml_pipeline.config.clf_rf_auto:
                grid_search_model = self.RF_GridSearch()
                grid_search_model.fit(x_train, y_train)
                chosen_model = grid_search_model.best_estimator_
                print(chosen_model)
            else:
                manual_params = self.ml_pipeline.config.clf_svm_manual

            evalclf = Evaluation.Evaluation(self.ml_pipeline)
            evalclf.evaluate_and_save_results(chosen_model, "rf")

            if self.ml_pipeline.config.clf_bagging_rf:
                n = self.ml_pipeline.config.clf_bag_rf_n
                evalclf.evaluate_bagging_model(chosen_model, n, "rf_bagging")


    def apply_lr(self):

        if self.ml_pipeline.config.clf_lr_flg:

            x_train = self.ml_pipeline.x_train
            y_train = self.ml_pipeline.y_train

            if self.ml_pipeline.config.clf_lr_auto:
                model = LogisticRegression(C=1.0, random_state=50, solver='liblinear')
                chosen_model = model.fit(x_train, y_train)
                print(chosen_model)
            else:
                manual_params = self.ml_pipeline.config.clf_svm_manual

            evalclf = Evaluation.Evaluation(self.ml_pipeline)
            evalclf.evaluate_and_save_results(chosen_model, "lr")

            if self.ml_pipeline.config.clf_bagging_lr:
                n = self.ml_pipeline.config.clf_bag_lr_n
                evalclf.evaluate_bagging_model(chosen_model, n, "lr_bagging")


    def apply_gnb(self):

        if self.ml_pipeline.config.clf_gnb_flg:

            x_train = self.ml_pipeline.x_train
            y_train = self.ml_pipeline.y_train

            if self.ml_pipeline.config.clf_gnb_auto:
                model = GaussianNB()
                chosen_model = model.fit(x_train, y_train)
                print(chosen_model)
            else:
                manual_params = self.ml_pipeline.config.clf_svm_manual

            evalclf = Evaluation.Evaluation(self.ml_pipeline)
            evalclf.evaluate_and_save_results(chosen_model, "gnb")

            if self.ml_pipeline.config.clf_bagging_gnb:
                n = self.ml_pipeline.config.clf_bag_gnb_n
                evalclf.evaluate_bagging_model(chosen_model, n, "gnb_bagging")


    def apply_mlp(self):

        if self.ml_pipeline.config.clf_mlp_flg:

            x_train = self.ml_pipeline.x_train
            y_train = self.ml_pipeline.y_train

            if self.ml_pipeline.config.clf_mlp_auto:
                grid_search_model = self.MLP_GridSearch()
                grid_search_model.fit(x_train, y_train)
                chosen_model = grid_search_model.best_estimator_
                print(chosen_model)
            else:
                manual_params = self.ml_pipeline.config.clf_svm_manual

            evalclf = Evaluation.Evaluation(self.ml_pipeline)
            evalclf.evaluate_and_save_results(chosen_model, "mlp")

            if self.ml_pipeline.config.clf_bagging_mlp:
                n = self.ml_pipeline.config.clf_bag_mlp_n
                evalclf.evaluate_bagging_model(chosen_model, n, "mlp_bagging")

    def SVM_GridSearch(self):
        random.seed(50)
        Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        gammas = [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
        kernel = ['rbf', 'poly', 'linear']
        param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernel}
        svm_clf = svm.SVC(probability=True)
        clf = GridSearchCV(svm_clf, param_grid, cv=5, n_jobs=-1, scoring='f1', verbose=3)
        return clf

    def MLP_GridSearch(self):
        parameter_space = {
            'hidden_layer_sizes': [(5, 5, 5), (20, 30, 50), (50, 50, 50), (50, 100, 50), (100,), (100, 100, 100),
                                   (5, 2)], 'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']}
        mlp = MLPClassifier(max_iter=1000, random_state=50)
        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5, scoring='f1', verbose=2)
        return clf

    def RF_GridSearch(self):
        n_estimators = [int(x) for x in np.linspace(start=2, stop=100, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                                       random_state=50, n_jobs=-1)
        return rf_random

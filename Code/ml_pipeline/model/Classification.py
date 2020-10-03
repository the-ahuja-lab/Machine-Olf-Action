import os
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pickle
import MLJobConfig as mlconfig

import MLPipeline
import AppConfig as app_config
import ml_pipeline.utils.Helper as helper
import Evaluation

DATA_FLD_NAME = app_config.CLF_FLD_NAME
DATA_FILE_NAME_PRFX = app_config.CLF_FLD_PREFIX


class Classification:

    def __init__(self, ml_pipeline: MLPipeline):

        self.ml_pipeline = ml_pipeline
        self.jlogger = self.ml_pipeline.jlogger

        self.jlogger.info(
            "Inside Classification initialization with status {}".format(self.ml_pipeline.status))
        step5 = os.path.join(self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME)
        os.makedirs(step5, exist_ok=True)

        if self.ml_pipeline.status == app_config.STEP4_STATUS:  # resuming at step 5
            self.apply_on_all_fg()

    def apply_on_all_fg(self):

        if self.ml_pipeline.config.fg_padelpy_flg:
            self.jlogger.info("Started classification of preprocessed PaDEL features")
            job_fld_path = self.ml_pipeline.job_data['job_fld_path']
            pp_padel_fld_path = os.path.join(
                *[job_fld_path, app_config.TEMP_TTS_FLD_NAME, app_config.FG_PADEL_FLD_NAME])

            padel_xtrain_fp = os.path.join(pp_padel_fld_path, app_config.TEMP_XTRAIN_FNAME)
            padel_ytrain_fp = os.path.join(pp_padel_fld_path, app_config.TEMP_YTRAIN_FNAME)
            padel_xtest_fp = os.path.join(pp_padel_fld_path, app_config.TEMP_XTEST_FNAME)
            padel_ytest_fp = os.path.join(pp_padel_fld_path, app_config.TEMP_YTEST_FNAME)

            self.ml_pipeline.x_train = pd.read_csv(padel_xtrain_fp)
            self.ml_pipeline.y_train = pd.read_csv(padel_ytrain_fp)
            self.ml_pipeline.y_train = self.ml_pipeline.y_train.values.ravel()

            self.ml_pipeline.x_test = pd.read_csv(padel_xtest_fp)
            self.ml_pipeline.y_test = pd.read_csv(padel_ytest_fp)
            self.ml_pipeline.y_test = self.ml_pipeline.y_test.values.ravel()

            # folder path to save output of preprocessed padel features classification data
            clf_padel_fld_path = os.path.join(*[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME,
                                                app_config.FG_PADEL_FLD_NAME])
            self.ml_pipeline.fg_clf_fld_path = clf_padel_fld_path
            os.makedirs(self.ml_pipeline.fg_clf_fld_path, exist_ok=True)

            self.apply_classification_models()

        if self.ml_pipeline.config.fg_mordered_flg:
            self.jlogger.info("Started classification of preprocessed mordred features")

            job_fld_path = self.ml_pipeline.job_data['job_fld_path']
            pp_mordred_fld_path = os.path.join(
                *[job_fld_path, app_config.TEMP_TTS_FLD_NAME, app_config.FG_MORDRED_FLD_NAME])
            mordred_xtrain_fp = os.path.join(pp_mordred_fld_path, app_config.TEMP_XTRAIN_FNAME)
            mordred_ytrain_fp = os.path.join(pp_mordred_fld_path, app_config.TEMP_YTRAIN_FNAME)
            mordred_xtest_fp = os.path.join(pp_mordred_fld_path, app_config.TEMP_XTEST_FNAME)
            mordred_ytest_fp = os.path.join(pp_mordred_fld_path, app_config.TEMP_YTEST_FNAME)

            self.ml_pipeline.x_train = pd.read_csv(mordred_xtrain_fp)
            self.ml_pipeline.y_train = pd.read_csv(mordred_ytrain_fp)
            self.ml_pipeline.y_train = self.ml_pipeline.y_train.values.ravel()

            self.ml_pipeline.x_test = pd.read_csv(mordred_xtest_fp)
            self.ml_pipeline.y_test = pd.read_csv(mordred_ytest_fp)
            self.ml_pipeline.y_test = self.ml_pipeline.y_test.values.ravel()

            # folder path to save output of preprocessed mordred features classification data
            clf_mordred_fld_path = os.path.join(*[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME,
                                                  app_config.FG_MORDRED_FLD_NAME])

            self.ml_pipeline.fg_clf_fld_path = clf_mordred_fld_path
            os.makedirs(self.ml_pipeline.fg_clf_fld_path, exist_ok=True)

            self.apply_classification_models()

        updated_status = app_config.STEP5_STATUS

        job_oth_config_fp = self.ml_pipeline.job_data['job_oth_config_path']
        helper.update_job_status(job_oth_config_fp, updated_status)

        self.ml_pipeline.status = updated_status

        self.jlogger.info("Classification completed successfully")

    def apply_classification_models(self):
        self.jlogger.info("Inside Classification, Train Shape: {}".format(self.ml_pipeline.x_train.shape))
        self.jlogger.info("Inside Classification, Test Shape: {}".format(self.ml_pipeline.x_test.shape))

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
                cv_inner = KFold(n_splits=10, shuffle=True, random_state=43)
                cv_outer = KFold(n_splits=5, shuffle=True, random_state=43)
                # clf = GradientBoostingClassifier(n_estimators=50, random_state=None, max_depth=2)
                grid_search_model = self.gbm_grid_search()
                grid_search_model.cv = cv_inner
                grid_search_model.fit(x_train, y_train)
                chosen_model = grid_search_model.best_estimator_
                scores = cross_val_score(grid_search_model, x_train, y_train, scoring='f1', cv=cv_outer, n_jobs=1)
                self.jlogger.info(str(chosen_model))
                self.jlogger.info("MEAN F1 scores after nested CV ", mean(scores))
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
                cv_inner = KFold(n_splits=10, shuffle=True, random_state=43)
                cv_outer = KFold(n_splits=5, shuffle=True, random_state=43)
                # clf = ExtraTreesClassifier(n_estimators=200, random_state=42, max_depth=10, n_jobs=-1)
                grid_search_model = self.et_grid_search()
                grid_search_model.cv = cv_inner
                grid_search_model.fit(x_train, y_train)
                chosen_model = grid_search_model.best_estimator_
                scores = cross_val_score(grid_search_model, x_train, y_train, scoring='f1', cv=cv_outer, n_jobs=1)
                self.jlogger.info(str(chosen_model))
                self.jlogger.info("MEAN F1 scores after nested CV ", mean(scores))
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
            cv_inner = KFold(n_splits=10, shuffle=True, random_state=43)
            cv_outer = KFold(n_splits=5, shuffle=True, random_state=43)
            if self.ml_pipeline.config.clf_svm_auto:
                grid_search_model = self.SVM_GridSearch()
                grid_search_model.cv = cv_inner
                grid_search_model.fit(x_train, y_train)
                chosen_model = grid_search_model.best_estimator_
                scores = cross_val_score(grid_search_model, x_train, y_train, scoring='f1', cv=cv_outer, n_jobs=1)
                self.jlogger.info(str(chosen_model))
                self.jlogger.info("MEAN F1 scores after nested CV ",mean(scores))
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
                cv_inner = KFold(n_splits=10, shuffle=True, random_state=43)
                cv_outer = KFold(n_splits=5, shuffle=True, random_state=43)
                grid_search_model = self.RF_GridSearch()
                grid_search_model.cv = cv_inner
                grid_search_model.fit(x_train, y_train)
                chosen_model = grid_search_model.best_estimator_
                scores = cross_val_score(grid_search_model, x_train, y_train, scoring='f1', cv=cv_outer, n_jobs=1)
                self.jlogger.info(str(chosen_model))
                self.jlogger.info("MEAN F1 scores after nested CV ", mean(scores))
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
                self.jlogger.info(str(chosen_model))
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
                self.jlogger.info(str(chosen_model))
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
                cv_inner = KFold(n_splits=10, shuffle=True, random_state=43)
                cv_outer = KFold(n_splits=5, shuffle=True, random_state=43)
                grid_search_model = self.MLP_GridSearch()
                grid_search_model.cv = cv_inner
                grid_search_model.fit(x_train, y_train)
                chosen_model = grid_search_model.best_estimator_
                scores = cross_val_score(grid_search_model, x_train, y_train, scoring='f1', cv=cv_outer, n_jobs=1)
                self.jlogger.info(str(chosen_model))
                self.jlogger.info("MEAN F1 scores after nested CV ", mean(scores))
            else:
                manual_params = self.ml_pipeline.config.clf_svm_manual

            evalclf = Evaluation.Evaluation(self.ml_pipeline)
            evalclf.evaluate_and_save_results(chosen_model, "mlp")

            if self.ml_pipeline.config.clf_bagging_mlp:
                n = self.ml_pipeline.config.clf_bag_mlp_n
                evalclf.evaluate_bagging_model(chosen_model, n, "mlp_bagging")

    def SVM_GridSearch(self):
        random.seed(50)
        if self.ml_pipeline.config.clf_hyp_man_c_svm:
            Cs = self.ml_pipeline.config.svm_C
        if self.ml_pipeline.config.clf_hyp_man_gamma_svm:
            gammas = self.ml_pipeline.config.svm_gamma
        if self.ml_pipeline.config.clf_hyp_man_kernel_svm:
            kernel = self.ml_pipeline.config.svm_kernels
        else:
            Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10]
            gammas = [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
            kernel = ['rbf', 'poly', 'linear']
        param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernel}
        svm_clf = svm.SVC(probability=True)
        clf = GridSearchCV(svm_clf, param_grid, cv=5, n_jobs=-1, scoring='f1', verbose=3)
        return clf

    def MLP_GridSearch(self):
        if self.ml_pipeline.config.clf_hyp_man_layers_mlp:
            hidden_layers_mlp = self.ml_pipeline.config.mlp_hidden_layers_list
        else:
            hidden_layers_mlp = [(5, 5, 5), (20, 30, 50), (50, 50, 50), (50, 100, 50), (100,), (100, 100, 100),
                                   (5, 2)]
        if self.ml_pipeline.config.clf_hyp_man_activation_mlp:
            activation_mlp = self.ml_pipeline.config.mlp_activation
        else:
            activation_mlp = ['tanh', 'relu']
        if self.ml_pipeline.config.clf_hyp_man_solver_mlp:
            solvers = self.ml_pipeline.config.mlp_solver
        else:
            solvers = ['sgd', 'adam']
        if self.ml_pipeline.config.clf_hyp_man_alpha_mlp:
            alphas = self.ml_pipeline.config.clf_hyp_alphas
        else:
            alphas = [0.0001, 0.05, 0.001, 0.01]
        if self.ml_pipeline.config.clf_hyp_man_lr_rate_mlp:
            learning_rates = self.ml_pipeline.config.mlp_lr
        else:
            learning_rates = ['constant', 'adaptive']
        parameter_space = {
            'hidden_layer_sizes': hidden_layers_mlp, 'activation': activation_mlp,
            'solver': solvers,
            'alpha': alphas,
            'learning_rate': learning_rates}
        mlp = MLPClassifier(max_iter=1000, random_state=50)
        clf = GridSearchCV(mlp, parameter_space,  n_jobs=-1, cv=5, scoring='f1', verbose=2)
        return clf

    def RF_GridSearch(self):
        if self.ml_pipeline.config.clf_hyp_man_depth_oth_rf:
            estimators = self.ml_pipeline.config.clf_hyp_man_estimate_oth_rf
        else:
            estimators = 100
        if self.ml_pipeline.config.clf_hyp_man_estimate_oth_rf:
            depth = self.ml_pipeline.config.clf_hyp_man_depth_oth_rf
        else:
            depth = 110
        if self.ml_pipeline.config.clf_hyp_man_sample_split_rf:
            sample_split_size = self.ml_pipeline.config.rf_sample_spit
        else:
            sample_split_size = [2, 5, 10]
        if self.ml_pipeline.config.clf_hyp_man_sample_leaf_rf:
            min_sample_leaf = self.ml_pipeline.config.rf_leaf
        else:
            min_sample_leaf = [1, 2, 4]
        if self.ml_pipeline.config.clf_hyp_man_bootstrap_rf:
            boot_strap = self.ml_pipeline.config.rf_bootstrap
        else:
            boot_strap = [True, False]
        n_estimators = [int(x) for x in np.linspace(start=2, stop=estimators, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, depth, num=11)]
        max_depth.append(None)
        min_samples_split = sample_split_size
        min_samples_leaf = min_sample_leaf
        bootstrap = boot_strap
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                                       random_state=50, n_jobs=-1, scoring="f1")
        return rf_random

    def gbm_grid_search(self):
        if self.ml_pipeline.config.clf_hyp_man_estimate_oth_gbm:
            estimators = self.ml_pipeline.config.clf_hyp_man_estimate_oth_gbm
        else:
            estimators = 510
        if self.ml_pipeline.config.clf_hyp_man_depth_oth_gbm:
            depth_param = self.ml_pipeline.config.clf_hyp_man_depth_oth_gbm
        else:
            depth_param = 20
        n_estimators = [int(x) for x in np.arange(start=10, stop=estimators, step=10)]
        max_depth = [int(x) for x in np.arange(start=2, stop=depth_param, step=2)]
        param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth}
        gbm = GradientBoostingClassifier(random_state=50)
        clf = GridSearchCV(gbm, param_grid, cv=5, scoring='f1', verbose=3, n_jobs=-1)
        return clf

    def et_grid_search(self):
        if self.ml_pipeline.config.clf_hyp_man_estimate_params_et:
            estimators = self.ml_pipeline.config.config.clf_hyp_man_estimate_params_et
        else:
            estimators = 510
        if self.ml_pipeline.config.clf_hyp_man_depth_et:
            depth = self.ml_pipeline.config.clf_hyp_man_depth_et
        else:
            depth = 20
        n_estimators = [int(x) for x in np.arange(start=10, stop=estimators, step=10)]
        max_depth = [int(x) for x in np.arange(start=2, stop=depth, step=2)]
        param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth}
        et = ExtraTreesClassifier(random_state=42, n_jobs=-1)
        clf = GridSearchCV(et, param_grid, cv=5, scoring='f1', verbose=3)
        return clf

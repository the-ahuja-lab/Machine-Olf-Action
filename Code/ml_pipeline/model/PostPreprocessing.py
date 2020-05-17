import MLPipeline
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.decomposition import PCA


class PostPreprocessing:

    def __init__(self, ml_pipeline: MLPipeline):
        print("Inside Preprocessing initialization")
        self.ml_pipeline = ml_pipeline
        self.preprocess_data()

    def preprocess_data(self):
        self.handle_missing_values()
        self.handle_class_imbalance()
        self.perform_data_normalization()
        self.handle_low_variance_cols()
        self.handle_correlated_cols()
        self.perform_boruta_fs()
        self.perform_pca()

    def handle_missing_values(self):

        if self.ml_pipeline.config.pp_mv_imputation_flg:
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
                # TODO if not mean, handle other cases
                pass

            self.ml_pipeline.x_train = X_train
            self.ml_pipeline.x_test = X_test

            print("Inside preprocessing, after handling missing values, train shape ", X_train.shape)
            print("Inside preprocessing, after handling missing values, test shape ", X_test.shape)

    def handle_class_imbalance(self):

        if self.ml_pipeline.config.pp_climb_smote_flg:
            xtrain = self.ml_pipeline.x_train
            ytrain = self.ml_pipeline.y_train

            print('Inside SMOTE, Original dataset shape %s' % Counter(ytrain))
            cols = xtrain.columns
            sm = SMOTE(random_state=50)
            xtrain_new, ytrain_new = sm.fit_resample(xtrain, ytrain)

            xtrain_df = pd.DataFrame(xtrain_new, columns=cols)
            print('Inside SMOTE, Resampled dataset shape %s' % Counter(ytrain_new))

            self.ml_pipeline.x_train = xtrain_df
            self.ml_pipeline.y_train = ytrain_new

    def perform_data_normalization(self):

        if self.ml_pipeline.config.pp_normalization_flg:

            traindata = self.ml_pipeline.x_train
            testdata = self.ml_pipeline.x_test

            try:
                x = traindata.values  # returns a numpy array
            except:
                x = traindata

            if self.ml_pipeline.config.pp_normalization_mthd == 'min_max':
                print("Inside performing minmax normalization")
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

        else:
            # TODO if not minmax, handle other cases
            pass

    def handle_low_variance_cols(self):

        if self.ml_pipeline.config.pp_vt_flg:
            th = self.ml_pipeline.config.pp_vt_th

            data_normal = self.ml_pipeline.x_train
            test_normal = self.ml_pipeline.x_test

            selector = VarianceThreshold(th)
            selector.fit(data_normal)
            data_var_free = data_normal[data_normal.columns[selector.get_support(indices=True)]]
            test_var_free = test_normal[test_normal.columns[selector.get_support(indices=True)]]

            self.ml_pipeline.x_train = data_var_free
            self.ml_pipeline.x_test = test_var_free

            print("Inside preprocessing, after variance removal, train shape ", data_var_free.shape)
            print("Inside preprocessing, after variance removal, test shape ", test_var_free.shape)

    def handle_correlated_cols(self):
        if self.ml_pipeline.config.pp_cr_flg:
            th = self.ml_pipeline.config.pp_cr_th

            print("corr th ", th)

            traindata = self.ml_pipeline.x_train
            testdata = self.ml_pipeline.x_test

            corr_matrix = traindata.corr()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            to_drop = [column for column in upper.columns if any(upper[column] > th)]
            trainset = traindata.drop(traindata[to_drop], axis=1)
            testset = testdata.drop(testdata[to_drop], axis=1)

            self.ml_pipeline.x_train = trainset
            self.ml_pipeline.x_test = testset

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

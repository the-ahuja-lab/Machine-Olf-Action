import MLPipeline
import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from joblib import dump, load

DATA_FLD_NAME = "step2"
DATA_FILE_NAME_PRFX = "PP_"


class Preprocessing:

    def __init__(self, ml_pipeline: MLPipeline):
        print("Inside Preprocessing initialization")

        self.ml_pipeline = ml_pipeline

        if self.ml_pipeline.status == "feature_generation":  # resuming at step 2
            print(ml_pipeline.job_data)
            padel_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step1", "FG_Padel.csv")
            data, data_labels = self.read_data(padel_data_fp)
            self.ml_pipeline.data = data
            self.ml_pipeline.data_labels = data_labels

            self.preprocess_data()

    def read_data(self, filepath):  # Input data should have one 1 column "Activation_Status" 1 column named "Ligand"
        data = pd.read_csv(filepath)
        data_labels = data["Activation Status"]
        Ligand_names = data["Ligand"]
        data = data.drop("Activation Status", axis=1)
        data = data.drop("Ligand", axis=1)
        try:
            data = data.drop("Smiles", axis=1)
        except:
            print("Don't have Smile's Column")
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

        #TODO check if all above steps successful
        self.write_to_csv_and_update_status()

    def perform_column_pruning(self):
        if self.ml_pipeline.config.pp_mv_col_pruning_flg:
            th = self.ml_pipeline.config.pp_mv_col_pruning_th

            data = self.ml_pipeline.data

            data = data.replace(r'\s+', np.nan, regex=True)
            data[data == np.inf] = np.nan
            data = data.replace(r'^\s*$', np.nan, regex=True)

            NAN_fld_path = self.ml_pipeline.job_data['job_data_path']
            NAN_fld_path = os.path.join(NAN_fld_path, DATA_FLD_NAME)

            NAN_file_path = os.path.join(NAN_fld_path, DATA_FILE_NAME_PRFX + "NAN_values1.csv")

            data.isna().sum().to_csv(NAN_file_path,
                                     header=False)  # TODO check if sum or sum/length i.e. avg greater than threshold

            NAN_data = pd.read_csv(NAN_file_path, header=None)
            dropped = []
            for i in range(len(NAN_data)):
                if NAN_data.iloc[i][1] >= th:
                    dropped.append(NAN_data.iloc[i][0])
            data = data.drop(dropped, axis=1)

            #TODO save files related to dropped columns list
            dropped_cols_lst_path = os.path.join(NAN_fld_path, DATA_FILE_NAME_PRFX + "NAN_Dropped.txt")
            with open(dropped_cols_lst_path, 'w') as f:
                for item in dropped:
                    f.write("%s\n" % item)

            print("Dropped columns: ", len(dropped))
            print("Data shape after pruning NAN values: ", data.shape)

            self.ml_pipeline.data = data

    def perform_train_test_split(self):

        test_per = self.ml_pipeline.config.tts_test_per / 100
        data = self.ml_pipeline.data
        data_labels = self.ml_pipeline.data_labels

        x_train, x_test, y_train, y_test = train_test_split(data, data_labels.values, test_size=test_per,
                                                            random_state=100, stratify=data_labels.values)

        self.ml_pipeline.x_train = x_train
        self.ml_pipeline.x_test = x_test
        self.ml_pipeline.y_train = y_train
        self.ml_pipeline.y_test = y_test

        #TODO save train test split files
        fld_path = self.ml_pipeline.job_data['job_data_path']
        fld_path = os.path.join(fld_path, DATA_FLD_NAME)

        train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "init_train.csv")
        test_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "init_test.csv")

        x_train.to_csv(train_file_path, index=False)
        x_test.to_csv(test_file_path, index=False)

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

            fld_path = self.ml_pipeline.job_data['job_data_path']
            fld_path = os.path.join(fld_path, DATA_FLD_NAME)

            train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "mv_train.csv")
            test_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "mv_test.csv")

            X_train.to_csv(train_file_path, index=False)
            X_test.to_csv(test_file_path, index=False)

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

            fld_path = self.ml_pipeline.job_data['job_data_path']
            fld_path = os.path.join(fld_path, DATA_FLD_NAME)

            train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "smote_train.csv")

            xtrain_df.to_csv(train_file_path, index=False)

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

                fld_path = self.ml_pipeline.job_data['job_data_path']
                fld_path = os.path.join(fld_path, DATA_FLD_NAME)

                train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "normalize_train.csv")
                test_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "normalize_test.csv")

                train_normal.to_csv(train_file_path, index=False)
                test_normal.to_csv(test_file_path, index=False)

                # TODO also pickle the minmax scaler model
                mm_scaler_model_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "data_normalization.joblib")
                dump(min_max_scaler, mm_scaler_model_path)

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

            fld_path = self.ml_pipeline.job_data['job_data_path']
            fld_path = os.path.join(fld_path, DATA_FLD_NAME)

            train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "varaince_train.csv")
            test_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "variance_test.csv")

            data_var_free.to_csv(train_file_path, index=False)
            test_var_free.to_csv(test_file_path, index=False)

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

            print(trainset.columns)

            print("Inside preprocessing, after correlation check, train shape ", trainset.shape)
            print("Inside preprocessing, after correlation check, test shape ", testset.shape)

    def write_to_csv_and_update_status(self):
        # TODO write to csv and update status

        x_train = self.ml_pipeline.x_train
        x_test = self.ml_pipeline.x_test
        ytrain = self.ml_pipeline.y_train
        ytest = self.ml_pipeline.y_test

        # x_train['Activation Status'] = ytrain
        # x_test['Activation Status'] = ytest

        fld_path = self.ml_pipeline.job_data['job_data_path']
        fld_path = os.path.join(fld_path, DATA_FLD_NAME)

        train_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "train.csv")
        test_file_path = os.path.join(fld_path, DATA_FILE_NAME_PRFX + "test.csv")

        x_train.to_csv(train_file_path, index=False)
        x_test.to_csv(test_file_path, index=False)

        #update status
        self.ml_pipeline.status = "preprocessing"

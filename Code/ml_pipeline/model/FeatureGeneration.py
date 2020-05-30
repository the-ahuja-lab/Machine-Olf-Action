import os
import pandas as pd

from collections import OrderedDict
from csv import DictReader
from datetime import datetime
import shutil

# PaDELPy imports
from padelpy import from_smiles
from padelpy.wrapper import padeldescriptor

# mordred imports
from mordred import Calculator, descriptors
from rdkit import Chem

from ml_pipeline.settings import APP_STATIC
import MLPipeline
from AppConfig import app_config
import ml_pipeline.utils.Helper as helper

DATA_FLD_NAME = app_config['fg_fld_name']


class FeatureGeneration:

    def __init__(self, ml_pipeline: MLPipeline, is_train: bool):
        self.ml_pipeline = ml_pipeline
        self.jlogger = self.ml_pipeline.jlogger

        self.is_train = is_train

        self.jlogger.info(
            "Inside FeatureGeneration initialization with status {} and is_train as {}".format(self.ml_pipeline.status,
                                                                                               self.is_train))

        # call only when in training state - inorder to reuse code for feature generation when not in training state
        if self.is_train:
            step1 = os.path.join(self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME)
            os.makedirs(step1, exist_ok=True)

            if self.ml_pipeline.status == app_config["step0_status"]:  # resuming at step 1
                if self.ml_pipeline.data is None:  # resuming from stop
                    self.jlogger.info("Resuming from step 1 of read user data")
                    user_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], app_config['user_ip_fld_name'],
                                                app_config['user_ip_fname'])

                    # Assuming data is already validated
                    data = pd.read_csv(user_data_fp)
                    self.ml_pipeline.data = data

                self.generate_features_from_smiles()

    def generate_features_from_smiles(self):
        self.jlogger.info("Inside generate_features_from_smiles while is_train flag {}".format(self.is_train))

        data = self.ml_pipeline.data
        org_data = data.copy()

        # TODO - Can run below two methods in different threads
        padel_df = self.generate_features_using_padel()
        if self.is_train and padel_df is not None:
            self.write_padel_features_to_csv(padel_df)

        # resetting data after being used by padel feature generation
        self.ml_pipeline.data = org_data.copy()
        mordred_df = self.generate_features_using_mordered()
        if self.is_train and mordred_df is not None:
            self.write_mordred_features_to_csv(mordred_df)

        if self.is_train:
            updated_status = app_config["step1_status"]

            job_oth_config_fp = self.ml_pipeline.job_data['job_oth_config_path']
            helper.update_job_status(job_oth_config_fp, updated_status)

            self.ml_pipeline.status = updated_status

        self.jlogger.info("Feature generation completed successfully")

    def from_smiles_dir(self, smiles_dir: str, output_csv: str = None, descriptors: bool = True,
                        fingerprints: bool = False, timeout: int = None) -> OrderedDict:
        ''' from_smiles: converts SMILES (smi) files present in a directory to QSPR descriptors/fingerprints

        Args:
            smiles_dir (str): SMILES (smil) files containing directory path
            output_csv (str): if supplied, saves descriptors of all smiles in the folder to this CSV file
            descriptors (bool): if `True`, calculates descriptors
            fingerprints (bool): if `True`, calculates fingerprints
            timeout (int): maximum time, in seconds, for conversion

        Returns:
            OrderedDict: descriptors/fingerprint labels and values
        '''

        # TODO handle space in path - java is giving issue..as of now hardcoded path C:/all_jobs/
        print("old output_csv ", output_csv)

        # output_csv = os.path.join(*[APP_STATIC, "compound_dbs", "temp_op_padel.csv"])
        #
        # output_csv = "C:/all_jobs"
        #
        # print("new output_csv ", output_csv)

        # save_csv = True
        # if output_csv is None:
        #     save_csv = False
        #     output_csv = 'padel_op.csv'

        for attempt in range(1):
            try:
                padeldescriptor(
                    mol_dir=smiles_dir,
                    d_file=output_csv,
                    convert3d=True,
                    retain3d=True,
                    d_2d=descriptors,
                    d_3d=descriptors,
                    fingerprints=fingerprints,
                    sp_timeout=timeout
                )
                break
            except RuntimeError as exception:
                if attempt == 0:
                    print("Exception occured ", exception)
                    # raise RuntimeError(exception)
            else:
                continue

        with open(output_csv, 'r', encoding='utf-8') as desc_file:
            reader = DictReader(desc_file)
            rows = [row for row in reader]
        desc_file.close()

        return rows

    def padel_desc_from_smile(self, smile, temp_smi_fld_path):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        with open(os.path.join(temp_smi_fld_path, '{}.smi'.format(timestamp)), 'w') as smi_file:
            smi_file.write(smile)

        return timestamp

    def clean_padel_name_col(self, str_name):
        name = str_name[8:]
        return name

    def generate_features_using_padel(self):

        if self.ml_pipeline.config.fg_padelpy_flg:
            self.jlogger.info("Inside generate_features_using_padel method")

            # TODO handle harcoded path
            temp_smi_fld_path = os.path.join("C:/all_jobs/", "SMI_Files")
            if os.path.exists(temp_smi_fld_path):
                shutil.rmtree(temp_smi_fld_path)
            os.makedirs(temp_smi_fld_path, exist_ok=True)

            df = self.ml_pipeline.data

            df = df[['SMILES', 'CNAME', 'Activation Status']]

            df["File_Names"] = df['SMILES'].apply(self.padel_desc_from_smile, temp_smi_fld_path=temp_smi_fld_path)

            temp_op_padel_path = os.path.join(temp_smi_fld_path, "temp_op_padel.csv")

            desc = self.from_smiles_dir(temp_smi_fld_path, output_csv=temp_op_padel_path,
                                        timeout=1800)  # 30 mins timeout

            op_padel_df = pd.DataFrame(desc)

            op_padel_df['Fin_CName'] = op_padel_df['Name'].apply(self.clean_padel_name_col)

            mrg_df = pd.merge(df, op_padel_df, how='inner', left_on='File_Names', right_on='Fin_CName')
            mrg_fin_df = mrg_df.drop(['SMILES', 'File_Names', 'Name', 'Fin_CName'], axis=1)

            mrg_fin_df = mrg_fin_df.sort_values('CNAME')
            ligands = mrg_fin_df['CNAME']
            mrg_fin_df = mrg_fin_df.drop(['CNAME'], axis=1)
            mrg_fin_df['CNAME'] = ligands

            # print(mrg_fin_df.columns)

            self.ml_pipeline.data = mrg_fin_df

            # clean up smi files
            if os.path.exists(temp_smi_fld_path):
                shutil.rmtree(temp_smi_fld_path)

            return mrg_fin_df

        else:
            return None

    def generate_features_using_mordered(self):
        if self.ml_pipeline.config.fg_mordered_flg:
            self.jlogger.info("Inside generate_features_using_mordered method")
            data = self.ml_pipeline.data
            calc = Calculator(descriptors, ignore_3D=True)
            mols = [Chem.MolFromSmiles(smi) for smi in data["SMILES"]]
            df = calc.pandas(mols)  ## All features
            df["CNAME"] = data["CNAME"]
            df["Activation Status"] = data["Activation Status"]

            mordred_df = df.copy()
            self.ml_pipeline.data = mordred_df
            return mordred_df
        else:
            return None

    def write_padel_features_to_csv(self, df):
        padel_fld_path = os.path.join(*[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME,
                                        app_config["fg_padel_fld_name"]])
        os.makedirs(padel_fld_path, exist_ok=True)

        padel_file_path = os.path.join(padel_fld_path, app_config["fg_padel_fname"])
        df.to_csv(padel_file_path, index=False)

    def write_mordred_features_to_csv(self, df):
        mordred_fld_path = os.path.join(*[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME,
                                          app_config["fg_mordred_fld_name"]])
        os.makedirs(mordred_fld_path, exist_ok=True)

        mordred_file_path = os.path.join(mordred_fld_path, app_config["fg_mordred_fname"])
        df.to_csv(mordred_file_path, index=False)

import MLPipeline
import os
from padelpy import from_smiles
import pandas as pd

from collections import OrderedDict
from csv import DictReader
from datetime import datetime
import shutil

# PaDELPy imports
from padelpy.wrapper import padeldescriptor

from ml_pipeline.settings import APP_STATIC

DATA_FLD_NAME = "step1"
DATA_FILE_NAME_PRFX = "FG"


class FeatureGeneration:

    def __init__(self, ml_pipeline: MLPipeline, is_train):
        print("Inside FeatureGeneration initialization")

        self.ml_pipeline = ml_pipeline
        self.is_train = is_train

        if self.ml_pipeline.status == "read_data":  # resuming at step 1
            if self.ml_pipeline.data is None:  # resuming from stop
                print(ml_pipeline.job_data)
                user_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step0", "user_data.csv")
                data = pd.read_csv(user_data_fp)
                self.ml_pipeline.data = data

            self.generate_features_from_smiles()

    def generate_features_from_smiles(self):
        padel_df = self.generate_features_using_padel()
        if self.is_train and padel_df is not None:
            self.write_padel_to_csv(padel_df)

        self.generate_features_using_mordered()
        # TODO - Can run above two methods in different threads

        # TODO update status that this stage is completed
        # update status
        self.ml_pipeline.status = "feature_generation"

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

            #TODO handle harcoded path
            temp_smi_fld_path = os.path.join("C:/all_jobs/", "SMI_Files")
            if os.path.exists(temp_smi_fld_path):
                shutil.rmtree(temp_smi_fld_path)
            os.makedirs(temp_smi_fld_path, exist_ok=True)

            df = self.ml_pipeline.data

            df = df[['Smiles', 'Ligand', 'Activation Status']]

            df["File_Names"] = df['Smiles'].apply(self.padel_desc_from_smile, temp_smi_fld_path=temp_smi_fld_path)

            temp_op_padel_path = os.path.join(temp_smi_fld_path, "temp_op_padel.csv")

            desc = self.from_smiles_dir(temp_smi_fld_path, output_csv=temp_op_padel_path, timeout=1800) #30 mins timeout

            op_padel_df = pd.DataFrame(desc)

            op_padel_df['Fin_CName'] = op_padel_df['Name'].apply(self.clean_padel_name_col)

            mrg_df = pd.merge(df, op_padel_df, how='inner', left_on='File_Names', right_on='Fin_CName')
            mrg_fin_df = mrg_df.drop(['Smiles', 'File_Names', 'Name', 'Fin_CName'], axis=1)

            mrg_fin_df = mrg_fin_df.sort_values('Ligand')
            ligands = mrg_fin_df['Ligand']
            mrg_fin_df = mrg_fin_df.drop(['Ligand'], axis=1)
            mrg_fin_df['Ligand'] = ligands

            print(mrg_fin_df.columns)

            self.ml_pipeline.data = mrg_fin_df

            # clean up smi files
            if os.path.exists(temp_smi_fld_path):
                shutil.rmtree(temp_smi_fld_path)

            return mrg_fin_df

        else:
            # TODO Log
            pass
            return None

    def generate_features_using_mordered(self):
        # TODO Check Mordered
        pass

    def write_padel_to_csv(self, df):
        padel_fld_path = self.ml_pipeline.job_data['job_data_path']
        padel_fld_path = os.path.join(padel_fld_path, DATA_FLD_NAME)

        padel_file_path = os.path.join(padel_fld_path, DATA_FILE_NAME_PRFX + "_Padel.csv")
        df.to_csv(padel_file_path, index=False)

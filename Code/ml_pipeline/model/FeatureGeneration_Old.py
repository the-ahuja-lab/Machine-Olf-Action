import MLPipeline
import os
from padelpy import from_smiles
import pandas as pd

DATA_FLD_NAME = "step1"
DATA_FILE_NAME_PRFX = "FG"


class FeatureGeneration_Old:

    def __init__(self, ml_pipeline: MLPipeline):
        print("Inside FeatureGeneration initialization")

        self.ml_pipeline = ml_pipeline

        if self.ml_pipeline.status == "read_data":  # resuming at step 1
            if self.ml_pipeline.data is None:  # resuming from stop
                print(ml_pipeline.job_data)
                user_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step0", "user_data.csv")
                data = pd.read_csv(user_data_fp)
                self.ml_pipeline.data = data

            self.generate_features_from_smiles()

    def generate_features_from_smiles(self):
        self.generate_features_using_padel()
        self.generate_features_using_mordered()
        # TODO - Can run above two methods in different threads

        # TODO update status that this stage is completed

    def generate_features_using_padel(self):

        fg_padelpy_flg = self.ml_pipeline.config.fg_padelpy_flg

        if fg_padelpy_flg:
            # TODO Make it parallel - Reduce this time
            test1 = self.ml_pipeline.data
            for i in range(len(test1)):
                print(i)
                try:
                    temp = test1["Smiles"][i]
                    descriptors = from_smiles(temp)
                except RuntimeError:
                    temp = test1["Smiles"][i]
                    descriptors = from_smiles(temp, timeout=30)
                if i == 0:
                    df = pd.DataFrame(descriptors, columns=descriptors.keys(), index=[0])
                else:
                    df1 = pd.DataFrame(descriptors, columns=descriptors.keys(), index=[i])
                if i is 1:
                    ff = pd.concat([df, df1], axis=0)
                if i > 1:
                    ff = pd.concat([ff, df1], axis=0)
            ff = pd.concat([ff, test1['Ligand']], axis=1)

            self.ml_pipeline.data = ff

            padel_fld_path = self.ml_pipeline.job_data['job_data_path']
            padel_fld_path = os.path.join(padel_fld_path, DATA_FLD_NAME)

            padel_file_path = os.path.join(padel_fld_path, DATA_FILE_NAME_PRFX + "_Padel.csv")

            ff.to_csv(padel_file_path, index=False)
        else:
            # TODO Log
            pass

    def generate_features_using_mordered(self):
        # TODO Check Mordered
        pass

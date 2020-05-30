import os
import pandas as pd

import MLPipeline
import FeatureGeneration as fg
from CompoundSimilarity import CompoundSimilarity
from ml_pipeline.settings import APP_STATIC
import AppConfig as app_config
import ml_pipeline.utils.Helper as helper

DATA_FLD_NAME = app_config.TSG_FLD_NAME
TEST_FLD_NAME = app_config.TSG_TEST_FLD_NAME


class TestSetGeneration:

    def __init__(self, ml_pipeline: MLPipeline):
        self.ml_pipeline = ml_pipeline
        self.jlogger = self.ml_pipeline.jlogger

        self.jlogger.info("Inside TestSetGeneration initialization")

        step6 = os.path.join(self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME)
        os.makedirs(step6, exist_ok=True)

        if self.ml_pipeline.status in [app_config.STEP1_STATUS,
                                       app_config.STEP5_STATUS]:  # only 1st step completion required
            user_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step0", "user_data.csv")
            data = pd.read_csv(user_data_fp)
            self.ml_pipeline.data = data

            self.pos_df = self.extract_pos_samples(data)

            self.search_similar_in_dbs()

    def extract_pos_samples(self, df):
        pos_df = df[df['Activation Status'] == 1]  # filtering for only positive samples
        pos_df = pos_df[['SMILES', 'CNAME']]
        pos_df = pos_df.drop_duplicates().reset_index(drop=True)  # drop duplicates if any
        return pos_df

    def search_similar_in_dbs(self):
        self.search_imppat()
        self.search_foodb()
        self.search_chebi()
        self.search_hmdb()

        updated_status = app_config.STEP6_STATUS

        job_oth_config_fp = self.ml_pipeline.job_data['job_oth_config_path']
        helper.update_job_status(job_oth_config_fp, updated_status)

        self.ml_pipeline.status = updated_status

        self.jlogger.info("Test set generation completed successfully")

    def search_imppat(self):

        if self.ml_pipeline.config.db_imppat_flg:
            compound_db_fld = os.path.join(APP_STATIC, "compound_dbs")
            imppat_db_fp = os.path.join(compound_db_fld, "imppat_2020_03_16.csv")
            self.jlogger.info("Imppat DB File Path {}".format(imppat_db_fp))

            db_df = pd.read_csv(imppat_db_fp, encoding="ISO-8859-1")

            c_sim_obj = CompoundSimilarity(self.pos_df, db_df)
            user_ip_fps, db_fps = c_sim_obj.calculate_fps_of_all_compounds()

            self.calculate_all_similarities(c_sim_obj, db_fps, "imppat")

    def search_foodb(self):

        if self.ml_pipeline.config.db_foodb_flg:
            compound_db_fld = os.path.join(APP_STATIC, "compound_dbs")
            food_db_fp = os.path.join(compound_db_fld, "foodb_final_2020_03_21.csv")

            self.jlogger.info("FoodDB DB File Path {}".format(food_db_fp))

            db_df = pd.read_csv(food_db_fp, encoding="ISO-8859-1")

            c_sim_obj = CompoundSimilarity(self.pos_df, db_df)
            user_ip_fps, db_fps = c_sim_obj.calculate_fps_of_all_compounds()

            self.calculate_all_similarities(c_sim_obj, db_fps, "foodb")

    def search_chebi(self):

        if self.ml_pipeline.config.db_chebi_flg:
            compound_db_fld = os.path.join(APP_STATIC, "compound_dbs")
            chebi_db_fp = os.path.join(compound_db_fld, "chebi_ds_2020_03_09.csv")
            self.jlogger.info("Chebi DB File Path {}".format(chebi_db_fp))

            db_df = pd.read_csv(chebi_db_fp, encoding="ISO-8859-1")

            c_sim_obj = CompoundSimilarity(self.pos_df, db_df)
            user_ip_fps, db_fps = c_sim_obj.calculate_fps_of_all_compounds()

            self.calculate_all_similarities(c_sim_obj, db_fps, "chebi")

    def search_hmdb(self):

        if self.ml_pipeline.config.db_hmdb_flg:
            compound_db_fld = os.path.join(APP_STATIC, "compound_dbs")
            hmdb_db_fp = os.path.join(compound_db_fld, "hmdb-2020-05-30.csv")

            self.jlogger.info("HMDB DB File Path {}".format(hmdb_db_fp))

            db_df = pd.read_csv(hmdb_db_fp, encoding="ISO-8859-1")

            c_sim_obj = CompoundSimilarity(self.pos_df, db_df)
            user_ip_fps, db_fps = c_sim_obj.calculate_fps_of_all_compounds()

            self.calculate_all_similarities(c_sim_obj, db_fps, "hmdb")

    def calculate_all_similarities(self, c_sim_obj, db_fps, db_name):
        # TODO calculate threshold automatically
        if self.ml_pipeline.config.sim_tanimoto_flg:
            self.calculate_and_save_sim_results(c_sim_obj, db_fps, db_name, "tanimoto", 0.8)

        if self.ml_pipeline.config.sim_dice_flg:
            self.calculate_and_save_sim_results(c_sim_obj, db_fps, db_name, "dice", 0.9)
        #
        if self.ml_pipeline.config.sim_cosine_flg:
            self.calculate_and_save_sim_results(c_sim_obj, db_fps, db_name, "cosine", 0.8)

        if self.ml_pipeline.config.sim_euclidean_flg:
            self.calculate_and_save_sim_results(c_sim_obj, db_fps, db_name, "euclidean", 0.8)

        if self.ml_pipeline.config.sim_manhattan_flg:
            self.calculate_and_save_sim_results(c_sim_obj, db_fps, db_name, "manhattan", 0.8)

        if self.ml_pipeline.config.sim_soergel_flg:
            self.calculate_and_save_sim_results(c_sim_obj, db_fps, db_name, "soergel", 0.95)

    def calculate_and_save_sim_results(self, c_sim_obj, db_fps, db_name, sim_metric, sim_threshold):

        all_novel_df, shortlisted_novel_df = c_sim_obj.check_similarity_using_fps(db_fps, sim_metric,
                                                                                  sim_threshold)

        fld_path = self.ml_pipeline.job_data['job_data_path']
        fld_path = os.path.join(*[fld_path, DATA_FLD_NAME, TEST_FLD_NAME])

        os.makedirs(fld_path, exist_ok=True)

        all_novel_fname = "all_matching_" + db_name + "_" + sim_metric + str(sim_threshold) + ".csv"
        shorlisted_novel_fname = "shortlisted_compounds_" + db_name + "_" + sim_metric + str(sim_threshold) + ".csv"

        all_novel_file_path = os.path.join(fld_path, all_novel_fname)
        shortlisted_file_path = os.path.join(fld_path, shorlisted_novel_fname)

        all_novel_df.to_csv(all_novel_file_path, index=False)
        shortlisted_novel_df.to_csv(shortlisted_file_path,
                                    index=False)

        # TODO change location of calling of this method, may be once automatic threshold calculation is done
        padel_fnmae = "padel_" + db_name + "_" + sim_metric + str(sim_threshold) + ".csv"
        self.generate_padel_features(shortlisted_novel_df, padel_fnmae)

        # TODO change location of calling of this method, may be once automatic threshold calculation is done
        mordred_fnmae = "mordred_" + db_name + "_" + sim_metric + str(sim_threshold) + ".csv"
        self.generate_mordred_features(shortlisted_novel_df, mordred_fnmae)

    def generate_padel_features(self, df, padel_file_name):
        fg_ml_pipeline = MLPipeline.MLPipeline(self.ml_pipeline.job_id)

        fg_ml_pipeline.data = df
        fg_obj = fg.FeatureGeneration(fg_ml_pipeline, is_train=False)
        padel_df = fg_obj.generate_features_using_padel()
        padel_df = padel_df.drop("Activation Status", axis=1)

        padel_raw_fld_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, app_config.FG_PADEL_FLD_NAME,
              app_config.TSG_RAW_FLD_NAME])
        padel_file_name = os.path.join(padel_raw_fld_path, padel_file_name)

        # make dir if not exists
        os.makedirs(padel_raw_fld_path, exist_ok=True)

        padel_df.to_csv(padel_file_name, index=False)

    def generate_mordred_features(self, df, mordred_file_name):
        fg_ml_pipeline = MLPipeline.MLPipeline(self.ml_pipeline.job_id)

        fg_ml_pipeline.data = df
        fg_obj = fg.FeatureGeneration(fg_ml_pipeline, is_train=False)
        mordred_df = fg_obj.generate_features_using_mordered()
        mordred_df = mordred_df.drop("Activation Status", axis=1)

        mordred_raw_fld_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, app_config.FG_MORDRED_FLD_NAME,
              app_config.TSG_RAW_FLD_NAME])
        mordred_file_name = os.path.join(mordred_raw_fld_path, mordred_file_name)

        # make dir if not exists
        os.makedirs(mordred_raw_fld_path, exist_ok=True)

        mordred_df.to_csv(mordred_file_name, index=False)

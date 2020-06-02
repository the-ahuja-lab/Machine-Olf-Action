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
        self.search_pubchem()
        self.search_custom_db()

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

            c_sim_obj = CompoundSimilarity(self.pos_df, db_df, self.jlogger)
            user_ip_fps, db_fps = c_sim_obj.calculate_fps_of_all_compounds()

            self.calculate_all_similarities(c_sim_obj, db_fps, "imppat")

    def search_foodb(self):

        if self.ml_pipeline.config.db_foodb_flg:
            compound_db_fld = os.path.join(APP_STATIC, "compound_dbs")
            food_db_fp = os.path.join(compound_db_fld, "foodb_2020_05_30.csv")

            self.jlogger.info("FoodDB DB File Path {}".format(food_db_fp))

            db_df = pd.read_csv(food_db_fp, encoding="ISO-8859-1")

            c_sim_obj = CompoundSimilarity(self.pos_df, db_df, self.jlogger)
            user_ip_fps, db_fps = c_sim_obj.calculate_fps_of_all_compounds()

            self.calculate_all_similarities(c_sim_obj, db_fps, "foodb")

    def search_chebi(self):

        if self.ml_pipeline.config.db_chebi_flg:
            compound_db_fld = os.path.join(APP_STATIC, "compound_dbs")
            chebi_db_fp = os.path.join(compound_db_fld, "chebi_ds_2020_03_09.csv")
            self.jlogger.info("Chebi DB File Path {}".format(chebi_db_fp))

            db_df = pd.read_csv(chebi_db_fp, encoding="ISO-8859-1")

            c_sim_obj = CompoundSimilarity(self.pos_df, db_df, self.jlogger)
            user_ip_fps, db_fps = c_sim_obj.calculate_fps_of_all_compounds()

            self.calculate_all_similarities(c_sim_obj, db_fps, "chebi")

    def search_hmdb(self):

        if self.ml_pipeline.config.db_hmdb_flg:
            compound_db_fld = os.path.join(APP_STATIC, "compound_dbs")
            hmdb_db_fp = os.path.join(compound_db_fld, "hmdb-2020-05-30.csv")

            self.jlogger.info("HMDB DB File Path {}".format(hmdb_db_fp))

            db_df = pd.read_csv(hmdb_db_fp, encoding="ISO-8859-1")

            c_sim_obj = CompoundSimilarity(self.pos_df, db_df, self.jlogger)
            user_ip_fps, db_fps = c_sim_obj.calculate_fps_of_all_compounds()

            self.calculate_all_similarities(c_sim_obj, db_fps, "hmdb")

    def search_custom_db(self):

        if self.ml_pipeline.config.db_pubchem_flg:
            compound_db_fld = "D:\IIITD\Semeseter\Winter2020\IP\pubchem_27052020\CUSTOM"

            for file in os.listdir(compound_db_fld):
                custom_db_fp = os.path.join(compound_db_fld, file)
                self.jlogger.info("Custom DB File Name {}".format(file))

                db_df = pd.read_csv(custom_db_fp, encoding="ISO-8859-1")

                c_sim_obj = CompoundSimilarity(self.pos_df, db_df, self.jlogger)
                user_ip_fps, db_fps = c_sim_obj.calculate_fps_of_all_compounds()

                custom_db_name = helper.change_ext(file, ".csv", "")

                self.calculate_all_similarities(c_sim_obj, db_fps, custom_db_name)

    def search_pubchem(self):

        if self.ml_pipeline.config.db_pubchem_flg:
            compound_db_fld = "D:\IIITD\Semeseter\Winter2020\IP\pubchem_27052020\PUBCHEM"

            fld_path = self.ml_pipeline.job_data['job_data_path']
            fld_path = os.path.join(*[fld_path, DATA_FLD_NAME, TEST_FLD_NAME])

            res_fld_path = os.path.join(fld_path, "pubchem")

            if not os.path.exists(res_fld_path):
                os.makedirs(res_fld_path, exist_ok=True)

            for file in os.listdir(compound_db_fld):
                pucbchem_db_part_fp = os.path.join(compound_db_fld, file)
                self.jlogger.info("Pubchem Part DB File Name {}".format(file))

                db_df = pd.read_csv(pucbchem_db_part_fp, encoding="ISO-8859-1")

                c_sim_obj = CompoundSimilarity(self.pos_df, db_df, self.jlogger)
                user_ip_fps, db_fps = c_sim_obj.calculate_fps_of_all_compounds()

                db_part_name = helper.change_ext(file, ".csv", "")

                self.calculate_db_part_similarity(c_sim_obj, db_fps, db_part_name, res_fld_path)

            self.combine_db_parts("pubchem", res_fld_path)
            
    def calculate_db_part_similarity(self, c_sim_obj, db_fps, db_name, res_fld_path):
        if self.ml_pipeline.config.sim_tanimoto_flg:
            self.calculate_db_part_sim_results(c_sim_obj, db_fps, db_name, "tanimoto", 0.8, res_fld_path)

        if self.ml_pipeline.config.sim_dice_flg:
            self.calculate_db_part_sim_results(c_sim_obj, db_fps, db_name, "dice", 0.9, res_fld_path)
        #
        if self.ml_pipeline.config.sim_cosine_flg:
            self.calculate_db_part_sim_results(c_sim_obj, db_fps, db_name, "cosine", 0.8, res_fld_path)

        if self.ml_pipeline.config.sim_euclidean_flg:
            self.calculate_db_part_sim_results(c_sim_obj, db_fps, db_name, "euclidean", 0.8, res_fld_path)

        if self.ml_pipeline.config.sim_manhattan_flg:
            self.calculate_db_part_sim_results(c_sim_obj, db_fps, db_name, "manhattan", 0.8, res_fld_path)

        if self.ml_pipeline.config.sim_soergel_flg:
            self.calculate_db_part_sim_results(c_sim_obj, db_fps, db_name, "soergel", 0.95, res_fld_path)

    def calculate_db_part_sim_results(self, c_sim_obj, db_fps, db_name, sim_metric, sim_threshold, res_fld_path):
        all_novel_df, shortlisted_novel_df = c_sim_obj.check_similarity_using_fps(db_fps, sim_metric,
                                                                                  sim_threshold)
        self.jlogger.info("Found {} similar compounds in {} using sim_metric {} and sim_threshold {}".format(
            len(shortlisted_novel_df), db_name, sim_metric, sim_threshold))

        all_novel_res_fld_path = os.path.join(*[res_fld_path, sim_metric, "all_novel"])
        shortlisted_novel_res_fld_path = os.path.join(*[res_fld_path, sim_metric, "shortlisted"])

        if not os.path.exists(all_novel_res_fld_path):
            os.makedirs(all_novel_res_fld_path, exist_ok=True)

        if not os.path.exists(shortlisted_novel_res_fld_path):
            os.makedirs(shortlisted_novel_res_fld_path, exist_ok=True)

        all_novel_fname = "all_matching_" + db_name + "_" + sim_metric + str(sim_threshold) + ".csv"
        shorlisted_novel_fname = "shortlisted_compounds_" + db_name + "_" + sim_metric + str(sim_threshold) + ".csv"

        all_novel_file_path = os.path.join(all_novel_res_fld_path, all_novel_fname)
        shortlisted_file_path = os.path.join(shortlisted_novel_res_fld_path, shorlisted_novel_fname)

        all_novel_df.to_csv(all_novel_file_path, index=False)
        shortlisted_novel_df.to_csv(shortlisted_file_path,
                                    index=False)

    def combine_db_parts(self, db_name, parts_fld_path):
        if self.ml_pipeline.config.sim_tanimoto_flg:
            sim_metric = "tanimoto"
            sim_th, all_novel_df, shortlisted_novel_df = self.combine_calculated_db_parts_sims(sim_metric,
                                                                                               parts_fld_path)
            self.save_results_generate_features(all_novel_df, shortlisted_novel_df, db_name, sim_metric, sim_th)

        if self.ml_pipeline.config.sim_dice_flg:
            sim_metric = "dice"
            sim_th, all_novel_df, shortlisted_novel_df = self.combine_calculated_db_parts_sims(sim_metric,
                                                                                               parts_fld_path)
            self.save_results_generate_features(all_novel_df, shortlisted_novel_df, db_name, sim_metric, sim_th)

        if self.ml_pipeline.config.sim_cosine_flg:
            sim_metric = "cosine"
            sim_th, all_novel_df, shortlisted_novel_df = self.combine_calculated_db_parts_sims(sim_metric,
                                                                                               parts_fld_path)
            self.save_results_generate_features(all_novel_df, shortlisted_novel_df, db_name, sim_metric, sim_th)

        if self.ml_pipeline.config.sim_euclidean_flg:
            sim_metric = "euclidean"
            sim_th, all_novel_df, shortlisted_novel_df = self.combine_calculated_db_parts_sims(sim_metric,
                                                                                               parts_fld_path)
            self.save_results_generate_features(all_novel_df, shortlisted_novel_df, db_name, sim_metric, sim_th)

        if self.ml_pipeline.config.sim_manhattan_flg:
            sim_metric = "manhattan"
            sim_th, all_novel_df, shortlisted_novel_df = self.combine_calculated_db_parts_sims(sim_metric,
                                                                                               parts_fld_path)
            self.save_results_generate_features(all_novel_df, shortlisted_novel_df, db_name, sim_metric, sim_th)

        if self.ml_pipeline.config.sim_soergel_flg:
            sim_metric = "soergel"
            sim_th, all_novel_df, shortlisted_novel_df = self.combine_calculated_db_parts_sims(sim_metric,
                                                                                               parts_fld_path)
            self.save_results_generate_features(all_novel_df, shortlisted_novel_df, db_name, sim_metric, sim_th)

    def combine_calculated_db_parts_sims(self, sim_metric, res_fld_path):
        all_novel_fld_path = os.path.join(res_fld_path, sim_metric, "all_novel")
        shortlisted_fld_path = os.path.join(res_fld_path, sim_metric, "shortlisted")

        fname = None

        df_all_novel_arr = []
        for file in os.listdir(all_novel_fld_path):
            fp = os.path.join(all_novel_fld_path, file)
            fname = file
            fl_df = pd.read_csv(fp)
            df_all_novel_arr.append(fl_df)

        sim_th = helper.infer_th_from_file_name(fname, sim_metric, ".csv")
        all_novel_df = pd.concat(df_all_novel_arr)

        df_all_shortlisted_arr = []
        for file in os.listdir(shortlisted_fld_path):
            fp = os.path.join(shortlisted_fld_path, file)
            fl_df = pd.read_csv(fp)
            df_all_shortlisted_arr.append(fl_df)

        shortlisted_novel_df = pd.concat(df_all_shortlisted_arr)

        return sim_th, all_novel_df, shortlisted_novel_df

    def calculate_all_similarities(self, c_sim_obj, db_fps, db_name):
        # TODO calculate threshold automatically
        if self.ml_pipeline.config.sim_tanimoto_flg:
            sim_metric = "tanimoto"
            sim_threshold = 0.8
            all_novel_df, shortlisted_novel_df = c_sim_obj.check_similarity_using_fps(db_fps, sim_metric,
                                                                                      sim_threshold)
            self.save_results_generate_features(all_novel_df, shortlisted_novel_df, db_name, sim_metric, sim_threshold)

        if self.ml_pipeline.config.sim_dice_flg:
            sim_metric = "dice"
            sim_threshold = 0.9
            all_novel_df, shortlisted_novel_df = c_sim_obj.check_similarity_using_fps(db_fps, sim_metric,
                                                                                      sim_threshold)
            self.save_results_generate_features(all_novel_df, shortlisted_novel_df, db_name, sim_metric, sim_threshold)
        #
        if self.ml_pipeline.config.sim_cosine_flg:
            sim_metric = "cosine"
            sim_threshold = 0.8
            all_novel_df, shortlisted_novel_df = c_sim_obj.check_similarity_using_fps(db_fps, sim_metric,
                                                                                      sim_threshold)
            self.save_results_generate_features(all_novel_df, shortlisted_novel_df, db_name, sim_metric, sim_threshold)

        if self.ml_pipeline.config.sim_euclidean_flg:
            sim_metric = "euclidean"
            sim_threshold = 0.8
            all_novel_df, shortlisted_novel_df = c_sim_obj.check_similarity_using_fps(db_fps, sim_metric,
                                                                                      sim_threshold)
            self.save_results_generate_features(all_novel_df, shortlisted_novel_df, db_name, sim_metric, sim_threshold)

        if self.ml_pipeline.config.sim_manhattan_flg:
            sim_metric = "manhattan"
            sim_threshold = 0.8
            all_novel_df, shortlisted_novel_df = c_sim_obj.check_similarity_using_fps(db_fps, sim_metric,
                                                                                      sim_threshold)
            self.save_results_generate_features(all_novel_df, shortlisted_novel_df, db_name, sim_metric, sim_threshold)

        if self.ml_pipeline.config.sim_soergel_flg:
            sim_metric = "soergel"
            sim_threshold = 0.95
            all_novel_df, shortlisted_novel_df = c_sim_obj.check_similarity_using_fps(db_fps, sim_metric,
                                                                                      sim_threshold)
            self.save_results_generate_features(all_novel_df, shortlisted_novel_df, db_name, sim_metric, sim_threshold)

    def save_results_generate_features(self, all_novel_df, shortlisted_novel_df, db_name, sim_metric,
                                       sim_threshold):

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
        if self.ml_pipeline.config.fg_padelpy_flg:
            fg_ml_pipeline = MLPipeline.MLPipeline(self.ml_pipeline.job_id)

            fg_ml_pipeline.data = df
            fg_obj = fg.FeatureGeneration(fg_ml_pipeline, is_train=False)
            padel_df = fg_obj.generate_features_using_padel()

            if not padel_df is None:
                padel_df = padel_df.drop("Activation Status", axis=1)

                padel_raw_fld_path = os.path.join(
                    *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, app_config.FG_PADEL_FLD_NAME,
                      app_config.TSG_RAW_FLD_NAME])
                padel_file_name = os.path.join(padel_raw_fld_path, padel_file_name)

                # make dir if not exists
                os.makedirs(padel_raw_fld_path, exist_ok=True)

                padel_df.to_csv(padel_file_name, index=False)

    def generate_mordred_features(self, df, mordred_file_name):
        if self.ml_pipeline.config.fg_mordered_flg:
            fg_ml_pipeline = MLPipeline.MLPipeline(self.ml_pipeline.job_id)

            fg_ml_pipeline.data = df
            fg_obj = fg.FeatureGeneration(fg_ml_pipeline, is_train=False)
            mordred_df = fg_obj.generate_features_using_mordered()

            if not mordred_df is None:
                mordred_df = mordred_df.drop("Activation Status", axis=1)

                mordred_raw_fld_path = os.path.join(
                    *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, app_config.FG_MORDRED_FLD_NAME,
                      app_config.TSG_RAW_FLD_NAME])
                mordred_file_name = os.path.join(mordred_raw_fld_path, mordred_file_name)

                # make dir if not exists
                os.makedirs(mordred_raw_fld_path, exist_ok=True)

                mordred_df.to_csv(mordred_file_name, index=False)

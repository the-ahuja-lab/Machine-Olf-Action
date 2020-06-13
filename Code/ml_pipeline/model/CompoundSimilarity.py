import pandas as pd
import math

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from tqdm import tqdm

import ml_pipeline.utils.Logging as logging

logger = logging.logger


# TODO clean this code
# Remove global variables, change it to class variable
# Fingerprints can be calculated before hand and saved in a file so that no need to calculate on every job

class CompoundSimilarity:

    def __init__(self, pos_user_df, db_df, logger):
        self.pos_user_df = pos_user_df
        self.db_df = db_df
        self.jlogger = logger
        tqdm.pandas()

    def smile_to_fingerprints(self, smile):
        fps = None
        try:
            mol = Chem.MolFromSmiles(smile)
            fps = FingerprintMols.FingerprintMol(mol)
        except Exception as e:
            self.jlogger.debug(
                "Error from rdkit while calculating fingerprint from smile, skipping this smile {}".format(smile))
        return fps

    def measure_similarity(self, db_fps, sim_metric=DataStructs.TanimotoSimilarity, th=0.8):
        global user_ip_fps
        global db_cntr
        global fps_matches

        if db_cntr % 10000 == 0:
            self.jlogger.info("Completed checking similarity with {} compound of db".format(db_cntr))

        u_fps_cntr = 0

        if not db_fps is None:
            for u_fps in user_ip_fps:
                try:
                    if not u_fps is None:
                        sim = DataStructs.FingerprintSimilarity(u_fps, db_fps, metric=sim_metric)
                        if sim >= th:
                            if db_cntr in fps_matches:
                                fps_matches[db_cntr].append((u_fps_cntr, sim))
                            else:
                                fps_matches[db_cntr] = [(u_fps_cntr, sim)]
                    else:
                        self.jlogger.debug(
                            "User Finger print is unavailable, skipping this compound {}".format(u_fps_cntr))
                except Exception as e:
                    logger.exception(
                        "Error measuring similarity of compound db_cntr {} and u_fps_cntr {}".format(db_cntr,
                                                                                                     u_fps_cntr))
                    self.jlogger.debug(
                        "Error measuring similarity of compound db_cntr {} and u_fps_cntr {}".format(db_cntr,
                                                                                                     u_fps_cntr))
                u_fps_cntr += 1

            db_cntr += 1
        else:
            self.jlogger.debug("DB Finger print is unavailable, skipping this compound {}".format(db_cntr))

    def measure_similarity_custom(self, db_fps, sim_metric=DataStructs.TanimotoSimilarity, th=0.8):
        global user_ip_fps
        global db_cntr
        global fps_matches

        if db_cntr % 10000 == 0:
            self.jlogger.info("Completed checking similarity with {} compound of db".format(db_cntr))

        u_fps_cntr = 0

        if not db_fps is None:
            for u_fps in user_ip_fps:
                try:
                    if not u_fps is None:
                        sim = sim_metric(u_fps, db_fps)
                        if sim >= th:
                            if db_cntr in fps_matches:
                                fps_matches[db_cntr].append((u_fps_cntr, sim))
                            else:
                                fps_matches[db_cntr] = [(u_fps_cntr, sim)]
                    else:
                        self.jlogger.debug(
                            "User Finger print is unavailable, skipping this compound {}".format(u_fps_cntr))
                except Exception as e:
                    logger.exception(
                        "Error measuring similarity of compound db_cntr {} and u_fps_cntr {}".format(db_cntr,
                                                                                                     u_fps_cntr))
                    self.jlogger.debug(
                        "Error measuring similarity of compound db_cntr {} and u_fps_cntr {}".format(db_cntr,
                                                                                                     u_fps_cntr))
                u_fps_cntr += 1

            db_cntr += 1
        else:
            self.jlogger.debug("DB Finger print is unavailable, skipping this compound {}".format(db_cntr))

    def get_vars_for_sim_calc(self, fp1, fp2):
        # ref: https://github.com/rdkit/rdkit-orig/blob/master/rdkit/DataStructs/__init__.py

        sz1 = fp1.GetNumBits()
        sz2 = fp2.GetNumBits()

        if sz1 < sz2:
            fp2 = DataStructs.FoldFingerprint(fp2, sz2 // sz1)
        elif sz2 < sz1:
            fp1 = DataStructs.FoldFingerprint(fp1, sz1 // sz2)

        a = fp1.GetNumOnBits()
        b = fp2.GetNumOnBits()
        c = len(DataStructs.OnBitsInCommon(fp1, fp2))

        return a, b, c

    # TODO distance v/s similarity measure calculations
    def measure_euclidean_similarity(self, fp1, fp2):
        a, b, c = self.get_vars_for_sim_calc(fp1, fp2)
        dist = math.sqrt(a + b - 2 * c)
        sim = 1 / (1 + dist)
        return sim

    def measure_manhattan_similarity(self, fp1, fp2):
        a, b, c = self.get_vars_for_sim_calc(fp1, fp2)
        dist = a + b - 2 * c
        sim = 1 / (1 + dist)
        return sim

    def measure_soergel_similarity(self, fp1, fp2):
        a, b, c = self.get_vars_for_sim_calc(fp1, fp2)
        return 1 - (c / (a + b - c))

    def get_sim_metric_rdkit_mapping(self, metric):
        if metric == "dice":
            return DataStructs.DiceSimilarity
        elif metric == "cosine":
            return DataStructs.CosineSimilarity
        elif metric == "tanimoto":
            return DataStructs.TanimotoSimilarity
        else:
            self.jlogger.error("Matching none of the similarity measure present, falling back to tanimoto similarity")
            return DataStructs.TanimotoSimilarity

    def get_sim_metric_custom_mapping(self, metric):
        if metric == "euclidean":
            return self.measure_euclidean_similarity
        elif metric == "manhattan":
            return self.measure_manhattan_similarity
        elif metric == "soergel":
            return self.measure_soergel_similarity
        else:
            return None

    def create_novel_ligands_csv(self, fps_matches):
        data = []
        for db_row in fps_matches:
            #         print(hmdb_row)
            db_row_srs = self.db_df.iloc[db_row]
            db_mol = db_row_srs['CNAME']
            db_smile = db_row_srs['SMILES']

            for usr_mtchs in fps_matches[db_row]:
                user_row_srs = self.pos_user_df.iloc[usr_mtchs[0]]
                sim = usr_mtchs[1]

                user_row_name = user_row_srs['CNAME']
                user_row_smile = user_row_srs['SMILES']

                data.append([user_row_name, user_row_smile, db_mol, db_smile, sim])
        return data

    def calculate_fps_of_all_compounds(self):
        global fps_matches
        global user_ip_fps
        global db_cntr

        user_ip_fps = None
        fps_matches = {}
        db_cntr = 0

        self.db_df = self.db_df[['CNAME', 'SMILES']]

        user_ip_fps = self.pos_user_df['SMILES'].progress_apply(self.smile_to_fingerprints)
        db_fps = self.db_df['SMILES'].progress_apply(self.smile_to_fingerprints)

        self.jlogger.info("Done generating fingerprints, starting to measure similarity")

        return user_ip_fps, db_fps

    def check_similarity_using_fps(self, db_fps, sim_metric="tanimoto",
                                   sim_threshold=0.8):
        global fps_matches
        global db_cntr
        fps_matches = {}
        db_cntr = 0

        fng_sim_metric = self.get_sim_metric_custom_mapping(sim_metric)
        if fng_sim_metric != None:
            # print("Inside custom similarity measure")
            tmp = db_fps.apply(self.measure_similarity_custom, sim_metric=fng_sim_metric, th=sim_threshold)
        else:
            # print("Inside rdkit similarity measure")
            fng_sim_metric = self.get_sim_metric_rdkit_mapping(sim_metric)
            tmp = db_fps.apply(self.measure_similarity, sim_metric=fng_sim_metric, th=sim_threshold)

        data = self.create_novel_ligands_csv(fps_matches)

        novel_df = pd.DataFrame(data, columns=["Test_Compound", "Test_SMILES", "DB_Compound", "DB_SMILES",
                                               "Similarity_Value"])
        # novel_df.to_csv("All_Matching_" + db_name + "_" + sim_metric + str(sim_threshold) + ".csv", index=False)

        fin_novel_df = novel_df[["DB_Compound", "DB_SMILES"]].drop_duplicates(subset='DB_Compound', keep='first')

        fin_novel_df = fin_novel_df.rename(columns={'DB_Compound': 'CNAME', 'DB_SMILES': 'SMILES'})
        fin_novel_df = fin_novel_df.reset_index(drop=True)
        fin_novel_df = fin_novel_df.sort_values('CNAME')
        fin_novel_df["Activation Status"] = ['?'] * len(fin_novel_df)

        # fin_novel_df.to_csv("Shortlisted_Compounds_" + db_name + "_" + sim_metric + str(sim_threshold) + ".csv",
        #                     index=False)

        return novel_df, fin_novel_df
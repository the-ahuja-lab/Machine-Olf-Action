class MLJobConfig:
    def empty(self):
        print("Inside MLJobConfig")
        self.fg_padelpy_flg = None
        self.fg_mordered_flg = None

        self.tts_train_per = None
        self.tts_test_per = None

        self.pp_mv_col_pruning_flg = None
        self.pp_mv_col_pruning_th = None
        self.pp_mv_imputation_flg = None
        self.pp_mv_imputation_mthd = None

        self.pp_climb_smote_flg = None

        self.pp_vt_flg = None
        self.pp_vt_th = None

        self.pp_cr_flg = None
        self.pp_cr_th = None

        self.pp_normalization_flg = None
        self.pp_normalization_mthd = None

        self.fs_boruta_flg = None

        self.fe_pca_flg = None
        self.fe_pca_energy = None

        self.clf_gbm_flg = None
        self.clf_gbm_auto = None
        self.clf_bagging_gbm = None
        self.clf_bag_gbm_n = None

        self.clf_svm_flg = None
        self.clf_svm_auto = None
        self.clf_bagging_svm = None
        self.clf_bag_svm_n = None

        self.clf_rf_flg = None
        self.clf_rf_auto = None
        self.clf_bagging_rf = None
        self.clf_bag_rf_n = None

        self.clf_lr_flg = None
        self.clf_lr_auto = None
        self.clf_bagging_lr = None
        self.clf_bag_lr_n = None

        self.clf_gnb_flg = None
        self.clf_gnb_auto = None
        self.clf_bagging_gnb = None
        self.clf_bag_gnb_n = None

        self.clf_et_flg = None
        self.clf_et_auto = None
        self.clf_bagging_et = None
        self.clf_bag_et_n = None

        self.clf_mlp_flg = None
        self.clf_mlp_auto = None
        self.clf_bagging_mlp = None
        self.clf_bag_mlp_n = None

        self.cv_3fold_flg = None
        self.cv_5fold_flg = None
        self.cv_loocv_flg = None

        self.db_hmdb_flg = None
        self.db_foodb_flg = None
        self.db_imppat_flg = None
        self.db_chebi_flg = None

        self.sim_tanimoto_flg = None
        self.sim_dice_flg = None
        self.sim_cosine_flg = None
        self.sim_euclidean_flg = None
        self.sim_manhattan_flg = None
        self.sim_soergel_flg = None

    def __init__(self, config=None):
        if config is None:
            self.empty()
        else:
            print("Inside MLJobConfig with config")
            self.fg_padelpy_flg = config.fg_padelpy_flg
            self.fg_mordered_flg = config.fg_mordered_flg

            self.tts_train_per = config.tts_train_per
            self.tts_test_per = config.tts_test_per

            self.pp_mv_col_pruning_flg = config.pp_mv_col_pruning_flg
            self.pp_mv_col_pruning_th = config.pp_mv_col_pruning_th
            self.pp_mv_imputation_flg = config.pp_mv_imputation_flg
            self.pp_mv_imputation_mthd = "mean"
            self.pp_climb_smote_flg = config.pp_climb_smote_flg
            self.pp_vt_flg = config.pp_vt_flg
            self.pp_vt_th = config.pp_vt_th
            self.pp_cr_flg = config.pp_cr_flg
            self.pp_cr_th = config.pp_cr_th
            self.pp_normalization_flg = config.pp_normalization_flg
            self.pp_normalization_mthd = "min_max"

            self.fs_boruta_flg = config.fs_boruta_flg

            self.fe_pca_flg = config.fe_pca_flg
            self.fe_pca_energy = config.fe_pca_energy

            self.clf_gbm_flg = config.clf_svm_flg
            self.clf_gbm_auto = True
            self.clf_bagging_gbm = config.clf_bagging_gbm
            self.clf_bag_gbm_n = config.clf_bag_gbm_n

            self.clf_svm_flg = config.clf_svm_flg
            self.clf_svm_auto = True
            self.clf_bagging_svm = config.clf_bagging_svm
            self.clf_bag_svm_n = config.clf_bag_svm_n

            self.clf_rf_flg = config.clf_rf_flg
            self.clf_rf_auto = True
            self.clf_bagging_rf = config.clf_bagging_rf
            self.clf_bag_rf_n = config.clf_bag_rf_n

            self.clf_lr_flg = config.clf_lr_flg
            self.clf_lr_auto = True
            self.clf_bagging_lr = config.clf_bagging_lr
            self.clf_bag_lr_n = config.clf_bag_lr_n

            self.clf_gnb_flg = config.clf_gnb_flg
            self.clf_gnb_auto = True
            self.clf_bagging_gnb = config.clf_bagging_gnb
            self.clf_bag_gnb_n = config.clf_bag_gnb_n

            self.clf_et_flg = config.clf_et_flg
            self.clf_et_auto = True
            self.clf_bagging_et = config.clf_bagging_et
            self.clf_bag_et_n = config.clf_bag_et_n

            self.clf_mlp_flg = config.clf_mlp_flg
            self.clf_mlp_auto = True
            self.clf_bagging_mlp = config.clf_bagging_mlp
            self.clf_bag_mlp_n = config.clf_bag_mlp_n

            self.cv_3fold_flg = config.cv_3fold_flg
            self.cv_5fold_flg = config.cv_5fold_flg
            self.cv_loocv_flg = config.cv_loocv_flg

            self.db_hmdb_flg = config.db_hmdb_flg
            self.db_foodb_flg = config.db_foodb_flg
            self.db_imppat_flg = config.db_imppat_flg
            self.db_chebi_flg = config.db_chebi_flg

            self.sim_tanimoto_flg = config.sim_tanimoto_flg
            self.sim_dice_flg = config.sim_dice_flg
            self.sim_cosine_flg = config.sim_cosine_flg
            self.sim_euclidean_flg = config.sim_euclidean_flg
            self.sim_manhattan_flg = config.sim_manhattan_flg
            self.sim_soergel_flg = config.sim_soergel_flg

    def __repr__(self):
        attrs = vars(self)
        print('\n '.join("%s: %s" % item for item in attrs.items()))
        return ""

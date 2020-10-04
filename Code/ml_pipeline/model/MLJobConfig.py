class MLJobConfig:

    def empty(self):
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
        self.rf_sample_spit = None

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

        self.exp_lime_flg = None

        self.db_hmdb_flg = None
        self.db_foodb_flg = None
        self.db_imppat_flg = None
        self.db_chebi_flg = None
        self.db_pubchem_flg = None
        self.db_custom_flg = None

        self.sim_tanimoto_flg = None
        self.sim_dice_flg = None
        self.sim_cosine_flg = None
        self.sim_euclidean_flg = None
        self.sim_manhattan_flg = None
        self.sim_soergel_flg = None

        # Hyperparameters for SVM
        self.clf_hyp_man_gamma_svm = None
        self.clf_hyp_man_kernel_svm = None
        self.clf_hyp_man_c_p1_svm = None
        self.clf_hyp_man_c_p2_svm = None
        self.clf_hyp_man_c_p3_svm = None
        self.clf_hyp_man_c_p4_svm = None
        self.clf_hyp_man_c_p5_svm = None
        self.clf_hyp_man_c_p6_svm = None
        self.clf_hyp_man_oth_svm = None
        self.svm_C = None
        self.clf_hyp_man_gamma_p1_svm = None
        self.clf_hyp_man_gamma_p2_svm = None
        self.clf_hyp_man_gamma_p3_svm = None
        self.clf_hyp_man_gamma_p4_svm = None
        self.clf_hyp_man_gamma_p5_svm = None
        self.clf_hyp_man_gamma_p6_svm = None
        self.clf_hyp_man_gamma_oth_svm = None
        self.svm_gamma = None
        self.clf_hyp_man_kernel_p1_svm = None
        self.clf_hyp_man_kernel_p2_svm = None
        self.clf_hyp_man_kernel_p3_svm = None
        self.svm_kernels = None

        # Hyperparameters for Extra Tree
        self.clf_hyp_man_et = None
        self.clf_hyp_man_estimator_et = None
        self.clf_hyp_man_estimate_oth_et = None
        self.clf_hyp_man_depth_params_et = None

        # Hyperparameters for Random Forest
        self.clf_hyp_man_estimator_rf = None
        self.clf_hyp_man_depth_rf = None
        self.clf_hyp_man_features_rf = None
        self.clf_hyp_man_sample_split_rf = None
        self.clf_hyp_man_sample_leaf_rf = None
        self.clf_hyp_man_bootstrap_rf = None
        self.rf_leaf = None
        self.rf_bootstrap = None
        self.clf_bagging_rf = None

        # Hyperparameters for Gradient Boosting Machine
        self.clf_hyp_man_depth_gbm = None
        self.clf_hyp_man_estimate_oth_rf = None

        # Hyperparameters for MLP
        self.clf_hyp_man_activation_mlp = None
        self.clf_hyp_man_alpha_mlp = None
        self.clf_hyp_man_solver_mlp = None
        self.clf_hyp_man_layers_mlp = None
        self.clf_hyp_man_lr_rate_p1_mlp = None
        self.clf_hyp_man_layers_p1_mlp = None
        self.clf_hyp_man_alpha_p2_mlp = None
        self.clf_hyp_man_activation_p1_mlp = None
        self.clf_hyp_man_activation_p2_mlp = None
        self.clf_hyp_man_solver_p1_mlp = None
        self.clf_hyp_man_solver_p2_mlp = None
        self.clf_hyp_man_lr_rate_p1_mlp = None
        self.clf_hyp_man_lr_rate_p2_mlp = None
        self.mlp_hidden_layers_list = None
        self.clf_hyp_alphas = None
        self.mlp_lr = None
        self.mlp_solver = None
        self.mlp_activation = None

        #Hyperparameters for LR
        self.clf_hyp_man_lr = None
        self.clf_hyp_man_c_p1_lr = None
        self.clf_hyp_man_c_p2_lr = None
        self.clf_hyp_man_c_p3_lr =None
        self.clf_hyp_man_c_p4_lr = None
        self.clf_hyp_man_c_p2_lr = None
        self.clf_hyp_man_oth_lr = None





    def __init__(self, config=None):
        if config is None:
            self.empty()
        else:
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

            self.clf_gbm_flg = config.clf_gbm_flg
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

            self.exp_lime_flg = config.exp_lime_flg

            self.db_hmdb_flg = config.db_hmdb_flg
            self.db_foodb_flg = config.db_foodb_flg
            self.db_imppat_flg = config.db_imppat_flg
            self.db_chebi_flg = config.db_chebi_flg
            self.db_pubchem_flg = config.db_pubchem_flg
            self.db_custom_flg = config.db_custom_flg

            self.sim_tanimoto_flg = config.sim_tanimoto_flg
            self.sim_dice_flg = config.sim_dice_flg
            self.sim_cosine_flg = config.sim_cosine_flg
            self.sim_euclidean_flg = config.sim_euclidean_flg
            self.sim_manhattan_flg = config.sim_manhattan_flg
            self.sim_soergel_flg = config.sim_soergel_flg

            # Hyperparameters for SVM
            self.clf_hyp_man_c_svm = config.clf_hyp_man_c_svm
            self.clf_hyp_man_gamma_svm = config.clf_hyp_man_gamma_svm
            self.clf_hyp_man_kernel_svm = config.clf_hyp_man_kernel_svm
            self.svm_C = []
            if config.clf_hyp_man_c_p1_svm:
                self.clf_hyp_man_c_p1_svm = 0.0001
                self.svm_C.append(self.clf_hyp_man_c_p1_svm)
            if config.clf_hyp_man_c_p2_svm:
                self.clf_hyp_man_c_p2_svm = 0.001
                self.svm_C.append(self.clf_hyp_man_c_p2_svm)
            if config.clf_hyp_man_c_p3_svm:
                self.clf_hyp_man_c_p3_svm = 0.01
                self.svm_C.append(self.clf_hyp_man_c_p3_svm)
            if config.clf_hyp_man_c_p4_svm:
                self.clf_hyp_man_c_p4_svm = 0.1
                self.svm_C.append(self.clf_hyp_man_c_p4_svm)
            if config.clf_hyp_man_c_p5_svm:
                self.clf_hyp_man_c_p5_svm = 1
                self.svm_C.append(self.clf_hyp_man_c_p5_svm)
            if config.clf_hyp_man_c_p6_svm:
                self.clf_hyp_man_c_p6_svm = 10
                self.svm_C.append(self.clf_hyp_man_c_p6_svm)
            if config.clf_hyp_man_oth_svm is not "":
                self.clf_hyp_man_oth_svm = config.clf_hyp_man_oth_svm
                self.svm_C.append(self.clf_hyp_man_oth_svm)
            self.svm_gamma = []
            if config.clf_hyp_man_gamma_p1_svm:
                self.clf_hyp_man_gamma_p1_svm = 0.0001
                self.svm_gamma.append(self.clf_hyp_man_gamma_p1_svm)
            if config.clf_hyp_man_gamma_p2_svm:
                self.clf_hyp_man_gamma_p2_svm = 0.001
                self.svm_gamma.append(self.clf_hyp_man_gamma_p2_svm)
            if config.clf_hyp_man_gamma_p3_svm:
                self.clf_hyp_man_gamma_p3_svm = 0.01
                self.svm_gamma.append(self.clf_hyp_man_gamma_p3_svm)
            if config.clf_hyp_man_gamma_p4_svm:
                self.clf_hyp_man_gamma_p4_svm = 0.1
                self.svm_gamma.append(self.clf_hyp_man_gamma_p4_svm)
            if config.clf_hyp_man_gamma_p5_svm:
                self.clf_hyp_man_gamma_p5_svm = 1
                self.svm_gamma.append(self.clf_hyp_man_gamma_p5_svm)
            if config.clf_hyp_man_gamma_p6_svm:
                self.clf_hyp_man_gamma_p6_svm = 10
                self.svm_gamma.append(self.clf_hyp_man_gamma_p6_svm)
            if config.clf_hyp_man_gamma_oth_svm:
                self.clf_hyp_man_gamma_oth_svm = config.clf_hyp_man_gamma_oth_svm
                self.svm_gamma.append(self.clf_hyp_man_gamma_oth_svm)

            self.svm_kernels = []
            self.clf_hyp_man_kernel_svm = config.clf_hyp_man_kernel_svm
            if config.clf_hyp_man_kernel_p1_svm:
                self.clf_hyp_man_kernel_p1_svm = "rbf"
                self.svm_kernels.append(self.clf_hyp_man_kernel_p1_svm)
            if config.clf_hyp_man_kernel_p2_svm:
                self.clf_hyp_man_kernel_p2_svm = "poly"
                self.svm_kernels.append(self.clf_hyp_man_kernel_p2_svm)
            if config.clf_hyp_man_kernel_p3_svm:
                self.clf_hyp_man_kernel_p3_svm = "linear"
                self.svm_kernels.append(self.clf_hyp_man_kernel_p3_svm)


            # Hyperparameters for Extra Tree
            self.clf_hyp_man_et = None
            self.clf_hyp_man_estimator_et = None


            #RF
            self.clf_hyp_man_estimator_rf = config.clf_hyp_man_estimator_rf
            self.clf_hyp_man_depth_rf = config.clf_hyp_man_depth_rf

            # Hyperparameters for Random Forest
            if config.clf_hyp_man_estimator_rf:
                self.clf_hyp_man_estimate_oth_rf = config.clf_hyp_man_estimate_oth_rf
            if config.clf_hyp_man_depth_rf:
                self.clf_hyp_man_depth_oth_rf = config.clf_hyp_man_depth_oth_rf
            if config.clf_hyp_man_features_rf:
                self.clf_hyp_man_features_oth_rf = config.clf_hyp_man_features_oth_rf

            self.clf_hyp_man_sample_split_rf = config.clf_hyp_man_sample_split_rf
            self.clf_hyp_man_sample_leaf_rf = config.clf_hyp_man_sample_leaf_rf
            self.rf_sample_spit = []
            if config.clf_hyp_man_sample_split_rf:
                self.clf_hyp_man_sample_split_p1_rf = 2
                self.rf_sample_spit.append(2)
            if config.clf_hyp_man_sample_leaf_rf:
                self.clf_hyp_man_sample_split_p2_rf = 5
                self.rf_sample_spit.append(5)
            if config.clf_hyp_man_bootstrap_rf:
                self.clf_hyp_man_sample_split_p3_rf = 10
                self.rf_sample_spit.append(10)

            self.rf_leaf = []
            if config.clf_hyp_man_sample_leaf_p1_rf:
                self.rf_leaf.append(1)
            if config.clf_hyp_man_sample_leaf_p2_rf:
                self.rf_leaf.append(2)
            if config.clf_hyp_man_sample_leaf_p3_rf:
                self.rf_leaf.append(4)

            self.rf_bootstrap = []
            if config.clf_hyp_man_bootstrap_p1_rf:
                self.rf_bootstrap.append(True)
            if config.clf_hyp_man_bootstrap_p2_rf:
                self.rf_bootstrap.append(False)

            # Hyperparameters for Gradient Boosting Machine
            self.clf_hyp_man_depth_gbm = config.clf_hyp_man_depth_gbm
            self.clf_hyp_man_estimate_oth_rf = config.clf_hyp_man_depth_gbm

            # Hyperparameters for MLP
            self.clf_hyp_man_activation_mlp = config.clf_hyp_man_activation_mlp
            self.clf_hyp_man_lr_rate_mlp = config.clf_hyp_man_lr_rate_mlp
            self.clf_hyp_man_solver_mlp = None
            self.clf_hyp_man_lr_rate_p1_mlp = None
            self.clf_hyp_man_layers_p1_mlp = None
            self.clf_hyp_man_alpha_p2_mlp = None
            self.clf_hyp_man_layers_mlp = config.clf_hyp_man_layers_mlp

            self.mlp_activation = []
            if config.clf_hyp_man_activation_p1_mlp:
                self.clf_hyp_man_activation_p1_mlp = "tanh"
                self.mlp_activation .append(self.clf_hyp_man_activation_p1_mlp)
            if config.clf_hyp_man_activation_p2_mlp:
                self.clf_hyp_man_activation_p2_mlp = "relu"
                self.mlp_activation .append(self.clf_hyp_man_activation_p2_mlp)

            self.mlp_solver = []
            if config.clf_hyp_man_solver_p1_mlp:
                self.clf_hyp_man_solver_p1_mlp = 'sgd'
                self.mlp_solver.append(self.clf_hyp_man_solver_p1_mlp )

            if config.clf_hyp_man_solver_p2_mlp:
                self.clf_hyp_man_solver_p2_mlp = 'adam'
                self.clf_hyp_man_solver_p2_mlp.append(self.clf_hyp_man_solver_p2_mlp)

            self.mlp_lr = []
            if config.clf_hyp_man_lr_rate_p1_mlp:
                self.clf_hyp_man_lr_rate_p1_mlp = 'constant'
                self.mlp_lr.append(self.clf_hyp_man_lr_rate_p1_mlp)
            if config.clf_hyp_man_lr_rate_p2_mlp:
                self.clf_hyp_man_lr_rate_p2_mlp = 'adaptive'
                self.mlp_lr.append(self.clf_hyp_man_lr_rate_p2_mlp)

            self.mlp_hidden_layers_list = []
            if config.clf_hyp_man_layers_p1_mlp:
                self.mlp_hidden_layers_list.append((5,5,5))
            if config.clf_hyp_man_layers_p2_mlp:
                self.mlp_hidden_layers_list.append((20,30,50))
            if config.clf_hyp_man_layers_p3_mlp:
                self.mlp_hidden_layers_list.append((50,50,50))
            if config.clf_hyp_man_layers_p4_mlp:
                self.mlp_hidden_layers_list.append((50,100,50))
            if config.clf_hyp_man_layers_p5_mlp:
                self.mlp_hidden_layers_list.append((100,100,100))
            if config.clf_hyp_man_layers_p6_mlp:
                self.mlp_hidden_layers_list.append((5,2))
            if config.clf_hyp_man_layers_p7_mlp:
                self.mlp_hidden_layers_list.append((100))

            self.clf_hyp_man_alpha_mlp = config.clf_hyp_man_alpha_mlp
            self.clf_hyp_alphas = []
            if config.clf_hyp_man_alpha_p2_mlp:
                self.clf_hyp_alphas.append(0.001)
            if config.clf_hyp_man_alpha_p3_mlp:
                self.clf_hyp_alphas.append(0.01)
            if config.clf_hyp_man_alpha_p4_mlp:
                self.clf_hyp_alphas.append(0.1)
            if config.clf_hyp_man_alpha_p5_mlp:
                self.clf_hyp_alphas.append(0.05)

            #ET
            self.clf_hyp_man_depth_et = config.clf_hyp_man_depth_et
            self.clf_hyp_man_depth_et = config.clf_hyp_man_depth_et
            if config.clf_hyp_man_estimator_et:
                self.clf_hyp_man_estimate_oth_et = config.clf_hyp_man_estimate_oth_et
            if config.clf_hyp_man_depth_et:
                self.clf_hyp_man_depth_oth_et = config.clf_hyp_man_depth_oth_et

            #GBM
            if config.clf_hyp_man_estimator_gbm:
                self.clf_hyp_man_estimate_oth_gbm = config.clf_hyp_man_estimator_gbm
            if config.clf_hyp_man_depth_gbm:
                self.clf_hyp_man_depth_oth_gbm = config.clf_hyp_man_depth_oth_gbm

            #LR
            self.clf_lr_list = []
            self.clf_hyp_man_lr = config.clf_hyp_man_lr
            if config.clf_hyp_man_c_p1_lr:
                self.clf_lr_list.append(0.0001)
            if config.clf_hyp_man_c_p2_lr:
                self.clf_lr_list.append(0.001)
            if config.clf_hyp_man_c_p3_lr:
                self.clf_lr_list.append(0.01)
            if config.clf_hyp_man_c_p4_lr:
                self.clf_lr_list.append(0.1)
            if config.clf_hyp_man_oth_lr is not "":
                self.clf_lr_list.append(config.clf_hyp_man_oth_lr)



    # TODO comment this if need to disable debug logging of config
    def __repr__(self):
        attrs = vars(self)
        obj_str = ('\n '.join("%s: %s" % item for item in attrs.items()))
        return obj_str


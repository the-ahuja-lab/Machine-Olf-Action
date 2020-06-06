$(document).ready(function() {

    $(document).on('change', "input[type=checkbox]", function() {
        var checkboxVal = (this.checked) ? 1 : 0;
        if (checkboxVal == 1) {
            $(this).prop("checked", true);
            $(this).val(1);
        } else {
            $(this).prop("checked", false);
            $(this).val(0);
        }
    });

});

$(document).ready(function() {
    $('#ml_job_frm').on('submit', function() {
        console.log('You submitted the form!');
        ckboxes = $('input[type="checkbox"]')

        $('input[type=checkbox]').each(function() {
            var checkboxVal = (this.checked) ? 1 : 0;
            console.log(this, checkboxVal)
            if (checkboxVal == 1) {
                $(this).prop("checked", true);
                $(this).val("on");
            } else {
                $(this).prop("checked", true);
                $(this).val("off");
            }
        });
    });
});

$(function() {
    $("#pp_cr_flg").click(function() {
        if ($(this).is(":checked")) {
            $("#dvCorrelation").show();
            $("#dvCorrelation :input").prop('required', true);
        } else {
            $("#dvCorrelation").hide();
            $("#dvCorrelation :input").prop('required', null);
        }
    });
});

$(function() {
    $("#pp_vt_flg").click(function() {
        if ($(this).is(":checked")) {
            $("#dvVariance").show();
            $("#dvVariance :input").prop('required', true);
        } else {
            $("#dvVariance").hide();
            $("#dvVariance :input").prop('required', null);
        }
    });
});


$(function() {
    $("#pp_mv_col_pruning_flg").click(function() {
        if ($(this).is(":checked")) {
            $("#prunecolumns").show();
            $("#prunecolumns :input").prop('required', true);
        } else {
            $("#prunecolumns").hide();
            $("#prunecolumns :input").prop('required', null);
        }
    });
});

$(function() {
    $("#fe_pca_flg").click(function() {
        if ($(this).is(":checked")) {
            $("#pca").show();
            $("#pca :input").prop('required', true);
        } else {
            $("#pca").hide();
            $("#pca :input").prop('required', null);
        }
    });
});

$(function() {
    $("#clf_svm_flg").click(function() {
        if ($(this).is(":checked")) {
            $("#svm").show();
        } else {
            $("#svm").hide();
        }
    });
});

$(function() {
    $("#clf_bagging_svm").click(function() {
        if ($(this).is(":checked")) {
            $("#dv_bag_svm_n").show();
            $("#dv_bag_svm_n :input").prop('required', true);
        } else {
            $("#dv_bag_svm_n").hide();
            $("#dv_bag_svm_n :input").prop('required', null);
        }
    });
});

$(function() {
    $("#clf_et_flg").click(function() {
        if ($(this).is(":checked")) {
            $("#extratrees").show();
        } else {
            $("#extratrees").hide();
        }
    });
});

$(function() {
    $("#clf_bagging_et").click(function() {
        if ($(this).is(":checked")) {
            $("#dv_bag_et_n").show();
            $("#dv_bag_et_n :input").prop('required', true);
        } else {
            $("#dv_bag_et_n").hide();
            $("#dv_bag_et_n :input").prop('required', null);
        }
    });
});

$(function() {
    $("#clf_lr_flg").click(function() {
        if ($(this).is(":checked")) {
            $("#LR").show();
        } else {
            $("#LR").hide();
        }
    });
});

$(function() {
    $("#clf_bagging_lr").click(function() {
        if ($(this).is(":checked")) {
            $("#dv_bag_lr_n").show();
            $("#dv_bag_lr_n :input").prop('required', true);
        } else {
            $("#dv_bag_lr_n").hide();
            $("#dv_bag_lr_n :input").prop('required', null);
        }
    });
});

$(function() {
    $("#clf_gnb_flg").click(function() {
        if ($(this).is(":checked")) {
            $("#GNB").show();
        } else {
            $("#GNB").hide();
        }
    });
});

$(function() {
    $("#clf_bagging_gnb").click(function() {
        if ($(this).is(":checked")) {
            $("#dv_bag_gnb_n").show();
            $("#dv_bag_gnb_n :input").prop('required', true);
        } else {
            $("#dv_bag_gnb_n").hide();
            $("#dv_bag_gnb_n :input").prop('required', null);
        }
    });
});

$(function() {
    $("#clf_gbm_flg").click(function() {
        if ($(this).is(":checked")) {
            $("#gbm").show();
        } else {
            $("#gbm").hide();
        }
    });
});

$(function() {
    $("#clf_bagging_gbm").click(function() {
        if ($(this).is(":checked")) {
            $("#dv_bag_gbm_n").show();
            $("#dv_bag_gbm_n :input").prop('required', true);
        } else {
            $("#dv_bag_gbm_n").hide();
            $("#dv_bag_gbm_n :input").prop('required', null);
        }
    });
});

$(function() {
    $("#clf_rf_flg").click(function() {
        if ($(this).is(":checked")) {
            $("#rforest").show();
        } else {
            $("#rforest").hide();
        }
    });
});

$(function() {
    $("#clf_bagging_rf").click(function() {
        if ($(this).is(":checked")) {
            $("#dv_bag_rf_n").show();
            $("#dv_bag_rf_n :input").prop('required', true);
        } else {
            $("#dv_bag_rf_n").hide();
            $("#dv_bag_rf_n :input").prop('required', null);
        }
    });
});

$(function() {
    $("#clf_mlp_flg").click(function() {
        if ($(this).is(":checked")) {
            $("#mlp").show();
        } else {
            $("#mlp").hide();
            show_simililarity_measures
        }
    });
});

$(function() {
    $("#clf_bagging_mlp").click(function() {
        if ($(this).is(":checked")) {
            $("#dv_bag_mlp_n").show();
            $("#dv_bag_mlp_n :input").prop('required', true);
        } else {
            $("#dv_bag_mlp_n").hide();
            $("#dv_bag_mlp_n :input").prop('required', null);
        }
    });
});

function reset_ml_job_form() {
    console.log("Reset Form")
    document.getElementById("ml_job_frm").reset();
}

function set_default_job_params() {
    console.log("Setting default job")
    $("#job_type").val("default_job");
    $("#job_description").val("Default OR1A1 Job");
}

function show_simililarity_measures() {

    hmdb = ($("#db_hmdb_flg").is(":checked"))
    foodb = ($("#db_foodb_flg").is(":checked"))
    imppat = ($("#db_imppat_flg").is(":checked"))
    chebi = ($("#db_chebi_flg").is(":checked"))
    pubchem = ($("#db_pubchem_flg").is(":checked"))

    if (hmdb || foodb || imppat || chebi || pubchem) {
        $("#sim_msrs_div").show();
    } else {
        $("#sim_msrs_div").hide();
    }

}
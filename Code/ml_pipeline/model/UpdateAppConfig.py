import os
import json

from AppConfig import fetch_app_config_path

import ml_pipeline.utils.Logging as logging

logger = logging.logger


def get_app_config():
    app_config_fld_path = fetch_app_config_path()
    app_config_fp = os.path.join(app_config_fld_path, "app_config.json")

    if os.path.exists(app_config_fp):
        with open(app_config_fp, mode='r') as f:
            all_app_configs = json.load(f)
            return all_app_configs
    else:
        return None


def update_app_config(all_app_configs):
    try:
        app_config_fld_path = fetch_app_config_path()
        app_config_fp = os.path.join(app_config_fld_path, "app_config.json")
        if not os.path.exists(app_config_fld_path):
            os.makedirs(app_config_fld_path, exist_ok=True)

        with open(app_config_fp, mode='w', encoding='utf-8') as f:
            json.dump(all_app_configs, f, ensure_ascii=False, indent=4)

        success = True
    except:
        success = False
        logger.exception("Exception occurred while updating app config ")

    return success


def check_if_valid_app_config(app_config_dict):
    # print("Config ", app_config_dict)

    error, app_config_dict = validate_error_form_fields(app_config_dict)
    return error, app_config_dict


def validate_error_form_fields(config_form_dict):
    logger.debug("Inside validate_error_form_fields after conversion dict {}".format(config_form_dict))

    ALL_VALID_ERRORS = []

    if not config_form_dict.get("pubchem_db_fld_path") is None \
            and config_form_dict.get("pubchem_db_fld_path").strip() != "" \
            and not os.path.exists(config_form_dict.get("pubchem_db_fld_path")):
        ALL_VALID_ERRORS.append("Pubchem folder path is not valid. Please provide a valid PubChem folder path.")

    if not config_form_dict.get("user_db_fld_path") is None \
            and config_form_dict.get("user_db_fld_path").strip() != "" \
            and not os.path.exists(config_form_dict.get("user_db_fld_path")):
        ALL_VALID_ERRORS.append(
            "Custom Database folder path is not valid. Please provide a valid custom database folder path.")

    if len(ALL_VALID_ERRORS) != 0:
        # if invalid return all the validation errors
        return True, ALL_VALID_ERRORS
    else:
        # if valid return all the configuration
        return False, config_form_dict

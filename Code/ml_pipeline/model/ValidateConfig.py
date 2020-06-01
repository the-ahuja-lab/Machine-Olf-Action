import ml_pipeline.utils.Logging as logging

logger = logging.logger

ALLOWED_USER_FILE_EXT = set(['csv'])


def validate_error_form_fields(config_form_dict):
    logger.debug("Inside validate_error_form_fields - user submitted dict {}".format(config_form_dict))
    config_user_dict = convert_user_dict_format(config_form_dict)
    logger.debug("Inside validate_error_form_fields after conversion dict {}".format(config_user_dict))

    ALL_VALID_ERRORS = []

    if not config_user_dict.get("fg_padelpy_flg") and not config_user_dict.get("fg_mordered_flg"):
        # if ALL_VALID_ERRORS["FG"] is None:
        #     ALL_VALID_ERRORS["FG"] = list()
        # ALL_VALID_ERRORS["FG"].append("Please select at-least one feature generation method from Padel or Mordered")
        ALL_VALID_ERRORS.append("Please select at-least one feature generation method from Padel or Mordered")

    if config_user_dict.get("tts_train_per") == 0:
        ALL_VALID_ERRORS.append("Please enter train split value greater than 0")

    if config_form_dict.get("tts_train_per") + config_form_dict.get("tts_test_per") != 100:
        ALL_VALID_ERRORS.append("Train and test split should add upto 100")

    # TODO add all other validations here

    if len(ALL_VALID_ERRORS) != 0:
        # if invalid return all the validation errors
        return True, ALL_VALID_ERRORS
    else:
        # if valid return all the configuration
        return False, config_user_dict


def check_if_valid_job_config(config_user_dict):
    # TODO - Perfrom validations on config like Validate if atleast one feature selection technique present
    # print("Config ", config_user_dict)

    error, config_user_dict = validate_error_form_fields(config_user_dict)
    return error, config_user_dict


def convert_user_dict_format(config_user_dict):
    """
    changes user supplied dictionary from html form to valid values python dictionary for downstream config input
    changes on/off flag to true/false and string numeric values python float values
    :param config_user_dict:
    :return:
    """
    for key, value in config_user_dict.items():
        # print(key, value)
        if value == "on":
            config_user_dict[key] = True
        elif value == "off":
            config_user_dict[key] = False
        elif value != None and value.strip != "":
            try:
                config_user_dict[key] = float(value)
            except:
                pass

    return config_user_dict


# TODO make this more secure than just checking extension, may be metadata checking
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_USER_FILE_EXT

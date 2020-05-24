import json
from collections import namedtuple

# from MLJobConfig import MLJobConfig
# from flask import url_for

# class ConfigPayload(object):
#     def __init__(self):
#         pass
#
#
# def as_config_payload(config):
#     cp = ConfigPayload()
#     cp.__dict__.update(config)
#     return cp

ALLOWED_EXTENSIONS = set(['csv'])

ALL_VALID_ERRORS = {}


def _json_object_hook(d):
    return namedtuple('x', d.keys())(*d.values())


def json2obj(data):
    return json.loads(data, object_hook=_json_object_hook)


def create_job_config_object(str):
    config = json2obj(str)
    print("Inside create_job_config_object ", config)
    return config


# def validate_create_job_config(str):
#     # print(str)
#     # config = json.loads(str, object_hook=as_config_payload)
#     config = json2obj(str)
#     # print(config)
#     # print(config.feature_generation.padelpy.is_enabled)
#     # print(type(config.train_test_split.train_per))
#     # validate_config(config)
#     return config

def validate_error_form_fields(config_form_dict):
    ALL_VALID_ERRORS = []
    config_user_dict = convert_user_dict_format(config_form_dict)

    print("Inside validate_form_fields ", config_user_dict)

    if not config_user_dict.get("fg_padelpy_flg") and not config_user_dict.get("fg_mordered_flg"):
        # if ALL_VALID_ERRORS["FG"] is None:
        #     ALL_VALID_ERRORS["FG"] = list()
        # ALL_VALID_ERRORS["FG"].append("Please select at-least one feature generation method from Padel or Mordered")
        ALL_VALID_ERRORS.append("Please select at-least one feature generation method from Padel or Mordered")

    if config_user_dict.get("tts_train_per") == 0:
        ALL_VALID_ERRORS.append("Please enter train split value greater than 0")

    # TODO add all other validations here

    if len(ALL_VALID_ERRORS) != 0:
        return True, ALL_VALID_ERRORS
    else:
        return False, config_user_dict


def check_if_valid_job_config(config_user_dict):
    # TODO - Perfrom validations on config like Validate if atleast one feature selection technique present
    # config = json2obj(str)
    print("Config ", config_user_dict)

    error, config_user_dict = validate_error_form_fields(config_user_dict)
    return error, config_user_dict


def convert_user_dict_format(config_user_dict):
    # print(type(config_user_dict))

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


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    # validate_create_job_config(url_for("static", filename = "ml_ip_config.json"))
    with open("../static/ml_ip_config.json") as f:
        json_str = f.read()
        check_if_valid_job_config(json_str)

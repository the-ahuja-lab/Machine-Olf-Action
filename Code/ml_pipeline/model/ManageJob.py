from ValidateConfig import create_job_config_object

from AppConfig import app_config
import os
from datetime import datetime
import json
# from shutil import copy

CONFIG_FLD_NAME = ".config"
DATA_FLD_NAME = "data"
RESULTS_FLD_NAME = "results"

CONFIG_FILE_NAME = "ml_ip_config.json"
OTH_CONFIG_FILE_NAME = "other_config.json"
STATUS_FILE_NAME = "status.txt"
LOG_FILE_NAME = "run.log"


def create_job(job_desc, job_config_json, user_uploaded_file, user_file):
    print("Inside Create Job with application root at ", user_uploaded_file)
    jobs_fld = app_config['jobs_folder']
    print(jobs_fld)
    # create a job id here

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    job_fld = os.path.join(jobs_fld, timestamp)

    os.makedirs(job_fld, exist_ok=True)

    job_id = timestamp

    config_fld_path = os.path.join(job_fld, CONFIG_FLD_NAME)
    data_fld_path = os.path.join(job_fld, DATA_FLD_NAME)
    results_fld_path = os.path.join(job_fld, RESULTS_FLD_NAME)

    # TODO remove this from here and add it to respective stage start location
    step0 = os.path.join(data_fld_path, "step0")
    os.makedirs(step0, exist_ok=True)

    # user_data_csv = os.path.join(user_uploaded_file, *["static", "user_data.csv"])
    #
    # copy(user_data_csv, step0)

    user_file.save(os.path.join(step0, "user_data.csv"))

    step1 = os.path.join(data_fld_path, "step1")
    os.makedirs(step1, exist_ok=True)

    # padel_csv = os.path.join(user_uploaded_file, *["static", "FG_Padel.csv"])
    #
    # copy(padel_csv, step1)

    step2 = os.path.join(data_fld_path, "step2")
    os.makedirs(step2, exist_ok=True)

    step3 = os.path.join(data_fld_path, "step3")
    os.makedirs(step3, exist_ok=True)

    step4 = os.path.join(data_fld_path, "step4")
    os.makedirs(step4, exist_ok=True)

    step5 = os.path.join(data_fld_path, "step5")
    os.makedirs(step5, exist_ok=True)

    step6 = os.path.join(data_fld_path, "step6")
    os.makedirs(step6, exist_ok=True)

    os.makedirs(config_fld_path, exist_ok=True)
    os.makedirs(data_fld_path, exist_ok=True)
    os.makedirs(results_fld_path, exist_ok=True)

    # json_config_file_path = os.path.join(config_fld_path, CONFIG_FILE_NAME)
    # with open(json_config_file_path, 'w', encoding='utf-8') as f:
    #     json_obj = json.loads(job_config_json)
    #     json.dump(json_obj, f, ensure_ascii=False, indent=4)

    json_config_file_path = os.path.join(config_fld_path, CONFIG_FILE_NAME)
    with open(json_config_file_path, 'w', encoding='utf-8') as f:
        json.dump(job_config_json, f, ensure_ascii=False, indent=4)

    status_file_path = os.path.join(config_fld_path, STATUS_FILE_NAME)
    with open(status_file_path, 'w', encoding='utf-8') as f:
        f.write("read_data")

    othr_job_config = {}
    othr_job_config['jd_text'] = job_desc
    othr_job_config['job_pid'] = ""
    othr_job_config['status'] = "read_data"

    oth_config_file_path = os.path.join(config_fld_path, OTH_CONFIG_FILE_NAME)
    with open(oth_config_file_path, 'w', encoding='utf-8') as f:
        json.dump(othr_job_config, f, ensure_ascii=False, indent=4)

    print("Job Successfully created with Job ID ", job_id)
    return job_id


def get_job_status(job_id):
    jobs_fld = app_config['jobs_folder']
    print(jobs_fld)
    # create a job id here

    # TODO read Job Status from Status file
    status = "read_data"

    return status


def get_job_details(job_id):
    job_details = {}
    jobs_fld = app_config['jobs_folder']
    job_id_fld = os.path.join(jobs_fld, job_id)

    job_details['job_fld_path'] = job_id_fld
    job_details['job_config_path'] = os.path.join(job_id_fld, CONFIG_FLD_NAME)
    job_details['job_data_path'] = os.path.join(job_id_fld, DATA_FLD_NAME)
    job_details['job_results_path'] = os.path.join(job_id_fld, RESULTS_FLD_NAME)

    config_fp = os.path.join(job_id_fld, CONFIG_FLD_NAME, CONFIG_FILE_NAME)
    status_fp = os.path.join(job_id_fld, CONFIG_FLD_NAME, STATUS_FILE_NAME)

    with open(config_fp) as f:
        json_str = f.read()
        job_config = create_job_config_object(json_str)

    with open(status_fp) as f:
        status = f.read()
        status = status.strip()
        # print("Status type, ", type(status))
        if status != None and status == '':
            status = None
        job_details['status'] = status

    return job_config, job_details


if __name__ == "__main__":
    with open("../static/ml_ip_config.json") as f:
        json_str = f.read()
        job_id = create_job("Default Job", json_str, "", "")
    get_job_details("1")

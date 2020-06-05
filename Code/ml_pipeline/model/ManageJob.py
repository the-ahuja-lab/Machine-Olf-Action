import os
from datetime import datetime
import json
from shutil import copy

from AppConfig import app_config
from ml_pipeline.settings import APP_ROOT
import ml_pipeline.utils.Helper as helper

import ml_pipeline.utils.Logging as logging

logger = logging.logger


def create_job(job_config_json, user_file, is_example_job):
    """
    Creates and initializes a new job id and create all the required folders of the job inside all jobs folder
    :param job_desc:
    :param job_config_json:
    :param user_file: user uploaded file object (if any)
    :param is_example_job: if an example job needs to be created
    :return: job id of newly created job
    """

    logger.debug("Inside create job with is_example_job flag as: {}".format(is_example_job))
    all_jobs_fld = app_config['jobs_folder']
    logger.debug("All jobs folder location: {}".format(all_jobs_fld))

    # create a job id here
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    job_id = timestamp

    # create the respective job id folder inside all jobs folder
    job_fld = os.path.join(all_jobs_fld, job_id)
    os.makedirs(job_fld, exist_ok=True)

    config_fld_path = os.path.join(job_fld, app_config['job_config_fld_name'])
    log_fld_path = os.path.join(*[job_fld, app_config['job_config_fld_name'], app_config['job_logs_fld_name']])
    data_fld_path = os.path.join(job_fld, app_config['job_data_fld_name'])
    results_fld_path = os.path.join(job_fld, app_config['job_results_fld_name'])

    step0 = os.path.join(data_fld_path, app_config['user_ip_fld_name'])
    os.makedirs(step0, exist_ok=True)

    # user_data_csv = os.path.join(APP_ROOT, *["static", "user_data.csv"])
    # copy(user_data_csv, step0)

    if not is_example_job:
        user_file.save(os.path.join(step0, app_config['user_ip_fname']))
    else:
        user_data_csv = os.path.join(APP_ROOT, *["static", "user_ip", "or1a1", app_config['user_ip_fname']])
        copy(user_data_csv, step0)

    # TODO remove this from here and add it to respective stage start location
    # step1 = os.path.join(data_fld_path, "step1")
    # os.makedirs(step1, exist_ok=True)

    # padel_csv = os.path.join(APP_ROOT, *["static", "FG_Padel.csv"])
    # copy(padel_csv, step1)

    # step2 = os.path.join(data_fld_path, "step2")
    # os.makedirs(step2, exist_ok=True)

    # step3 = os.path.join(data_fld_path, "step3")
    # os.makedirs(step3, exist_ok=True)

    # step4 = os.path.join(data_fld_path, "step4")
    # os.makedirs(step4, exist_ok=True)

    # step5 = os.path.join(data_fld_path, "step5")
    # os.makedirs(step5, exist_ok=True)

    # step6 = os.path.join(data_fld_path, "step6")
    # os.makedirs(step6, exist_ok=True)

    os.makedirs(config_fld_path, exist_ok=True)
    os.makedirs(log_fld_path, exist_ok=True)
    os.makedirs(data_fld_path, exist_ok=True)
    os.makedirs(results_fld_path, exist_ok=True)

    json_config_file_path = os.path.join(config_fld_path, app_config['job_config_fname'])
    with open(json_config_file_path, 'w', encoding='utf-8') as f:
        json.dump(job_config_json, f, ensure_ascii=False, indent=4)

    init_status = app_config["job_init_status"]  # TODO change it to "job created" instead of "read_data"

    # status_file_path = os.path.join(config_fld_path, app_config['job_status_fname'])
    # with open(status_file_path, 'w', encoding='utf-8') as f:
    #     f.write(init_status)

    othr_job_config = {}
    othr_job_config['job_id'] = job_id
    othr_job_config['jd_text'] = job_config_json["job_description"]  # TODO consider removing job description from here
    othr_job_config['job_pid'] = ""  # TODO set it later when spwanning new process from job
    othr_job_config['status'] = init_status
    othr_job_config['job_run_status'] = ""

    oth_config_file_path = os.path.join(config_fld_path, app_config['job_other_config_fname'])
    with open(oth_config_file_path, 'w', encoding='utf-8') as f:
        json.dump(othr_job_config, f, ensure_ascii=False, indent=4)

    logger.info("Job Successfully created with Job ID {}".format(job_id))
    return job_id


def get_job_details(job_id):
    """
    retrieves all job related configuration details
    :param job_id: job id whose details needs to be retrieved
    :return:
    tuple - job_config, job_details
    job_config - dictionary from user uploaded json, job_details - map with other job folder related params
    """
    job_details = {}
    jobs_fld = app_config['jobs_folder']
    job_id_fld = os.path.join(jobs_fld, job_id)

    job_details['job_fld_path'] = job_id_fld
    job_details['job_config_path'] = os.path.join(job_id_fld, app_config['job_config_fld_name'])
    job_details['job_data_path'] = os.path.join(job_id_fld, app_config['job_data_fld_name'])
    job_details['job_results_path'] = os.path.join(job_id_fld, app_config['job_results_fld_name'])
    job_details['job_log_path'] = os.path.join(
        *[job_id_fld, app_config['job_config_fld_name'], app_config['job_logs_fld_name']])
    job_details['job_oth_config_path'] = os.path.join(*[job_id_fld, app_config['job_config_fld_name'],
                                                        app_config['job_other_config_fname']])

    config_fp = os.path.join(job_id_fld, app_config['job_config_fld_name'], app_config['job_config_fname'])
    # status_fp = os.path.join(job_id_fld, app_config['job_config_fld_name'], app_config['job_status_fname'])
    status_fp = job_details['job_oth_config_path']

    # TODO consider adding user_config validation here too
    with open(config_fp) as f:
        json_str = f.read()
        job_config = helper.create_job_config_object(json_str)

    # TODO consider read status from otherconfig json instead of status.txt file
    with open(status_fp) as f:
        other_configs = json.load(f)
        status = other_configs['status']
        status = status.strip()
        if status == None and status == '':
            status = None
        job_details['status'] = status

    return job_config, job_details



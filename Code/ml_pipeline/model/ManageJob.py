import os
from datetime import datetime
import json
from shutil import copy

import AppConfig as app_config
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
    all_jobs_fld = app_config.ALL_JOBS_FOLDER
    logger.debug("All jobs folder location: {}".format(all_jobs_fld))

    # create a job id here
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    job_id = timestamp

    # create the respective job id folder inside all jobs folder
    job_fld = os.path.join(all_jobs_fld, job_id)
    os.makedirs(job_fld, exist_ok=True)

    config_fld_path = os.path.join(job_fld, app_config.JOB_CONFIG_FLD_NAME)
    log_fld_path = os.path.join(*[job_fld, app_config.JOB_CONFIG_FLD_NAME, app_config.JOB_LOGS_FLD_NAME])
    data_fld_path = os.path.join(job_fld, app_config.JOB_DATA_FLD_NAME)
    results_fld_path = os.path.join(job_fld, app_config.JOB_RESULTS_FLD_NAME)

    step0 = os.path.join(data_fld_path, app_config.USER_IP_FLD_NAME)
    os.makedirs(step0, exist_ok=True)

    if not is_example_job:
        user_file.save(os.path.join(step0, app_config.USER_IP_FNAME))
    else:
        user_data_csv = os.path.join(APP_ROOT, *["static", "user_ip", "or1a1", app_config.USER_IP_FNAME])
        copy(user_data_csv, step0)

    os.makedirs(config_fld_path, exist_ok=True)
    os.makedirs(log_fld_path, exist_ok=True)
    os.makedirs(data_fld_path, exist_ok=True)
    os.makedirs(results_fld_path, exist_ok=True)

    json_config_file_path = os.path.join(config_fld_path, app_config.JOB_CONFIG_FNAME)
    with open(json_config_file_path, 'w', encoding='utf-8') as f:
        json.dump(job_config_json, f, ensure_ascii=False, indent=4)

    init_status = app_config.JOB_INIT_STATUS

    othr_job_config = {}
    othr_job_config['job_id'] = job_id
    othr_job_config['jd_text'] = job_config_json["job_description"]
    othr_job_config['status'] = init_status
    othr_job_config['job_run_status'] = ""

    oth_config_file_path = os.path.join(config_fld_path, app_config.JOB_OTHER_CONFIG_FNAME)
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
    jobs_fld = app_config.ALL_JOBS_FOLDER
    job_id_fld = os.path.join(jobs_fld, job_id)

    job_details['job_fld_path'] = job_id_fld
    job_details['job_config_path'] = os.path.join(job_id_fld, app_config.JOB_CONFIG_FLD_NAME)
    job_details['job_data_path'] = os.path.join(job_id_fld, app_config.JOB_DATA_FLD_NAME)
    job_details['job_results_path'] = os.path.join(job_id_fld, app_config.JOB_RESULTS_FLD_NAME)
    job_details['job_log_path'] = os.path.join(
        *[job_id_fld, app_config.JOB_CONFIG_FLD_NAME, app_config.JOB_LOGS_FLD_NAME])
    job_details['job_oth_config_path'] = os.path.join(*[job_id_fld, app_config.JOB_CONFIG_FLD_NAME,
                                                        app_config.JOB_OTHER_CONFIG_FNAME])

    config_fp = os.path.join(job_id_fld, app_config.JOB_CONFIG_FLD_NAME, app_config.JOB_CONFIG_FNAME)
    status_fp = job_details['job_oth_config_path']

    # TODO consider adding user_config validation here too
    with open(config_fp) as f:
        json_str = f.read()
        job_config = helper.create_job_config_object(json_str)

    with open(status_fp) as f:
        other_configs = json.load(f)
        status = other_configs['status']
        status = status.strip()
        if status == None and status == '':
            status = None
        job_details['status'] = status

    return job_config, job_details



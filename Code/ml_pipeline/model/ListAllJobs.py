from AppConfig import app_config
import os
from datetime import datetime
import json


class MLJob:
    def __init__(self, job_id, job_status, job_desc, job_created_time):
        self.job_id = job_id
        self.job_status = job_status
        self.job_desc = job_desc
        self.job_created_time = job_created_time


class ListAllJobs:

    def __init__(self):
        self.all_jobs_fld = app_config['jobs_folder']

    def get_all_jobs(self):

        all_ml_jobs_listing = []

        all_job_flds = os.listdir(self.all_jobs_fld)

        for fld in all_job_flds:
            job_id_path = os.path.join(self.all_jobs_fld, fld)
            if os.path.isdir(job_id_path):  # if directory
                job_fld = os.path.join(self.all_jobs_fld, job_id_path)
                job_created_time = datetime.fromtimestamp(os.path.getctime(job_fld)).replace(microsecond=0)
                job_status = self.get_job_status(job_fld)
                job_desc = self.get_job_desc(job_fld)

                print("Job ID: ", fld)
                print("Job Created Time : ", job_created_time)
                print("Job Status: ", job_status)
                print("Job Description : ", job_desc)

                ml_job = MLJob(fld, job_status, job_desc, job_created_time)

                all_ml_jobs_listing.append(ml_job)

                all_ml_jobs_listing = self.sort_jobs(all_ml_jobs_listing, "created")

        return all_ml_jobs_listing

    def sort_jobs(self, all_ml_jobs_listing, sort_by):

        if sort_by == "created":
            all_ml_jobs_listing = sorted(all_ml_jobs_listing, key=lambda x: x.job_created_time, reverse=True)

        return all_ml_jobs_listing

    def get_job_status(self, job_id):
        jobs_fld = app_config['jobs_folder']
        print(jobs_fld)
        # create a job id here

        # TODO read Job Status from Status file
        oth_config_json_path = os.path.join(*[jobs_fld, job_id, ".config", "other_config.json"])
        print(oth_config_json_path)
        with open(oth_config_json_path) as f:
            json_str = f.read()
            othr_config = json.loads(json_str)

        print(othr_config)
        print(othr_config['status'])

        status = othr_config['status']

        return status

    def get_job_desc(self, job_id):
        jobs_fld = app_config['jobs_folder']
        print(jobs_fld)
        # create a job id here

        oth_config_json_path = os.path.join(*[jobs_fld, job_id, ".config", "other_config.json"])
        with open(oth_config_json_path) as f:
            json_str = f.read()
            othr_config = json.loads(json_str)

        print(othr_config)
        print(othr_config['jd_text'])

        # TODO read Job Status from Status file
        job_desc = othr_config['jd_text']

        return job_desc

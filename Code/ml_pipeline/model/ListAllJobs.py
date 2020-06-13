import os
from datetime import datetime
import json

import AppConfig as app_config
from ml_pipeline import running_jobs_details
import ml_pipeline.utils.Helper as helper


class MLJob:
    def __init__(self, job_id, job_last_status, job_desc, job_created_time, job_run_status):
        self.job_id = job_id
        self.job_last_status = job_last_status
        self.job_desc = job_desc
        self.job_created_time = job_created_time
        self.job_run_status = job_run_status


class ListAllJobs:

    def __init__(self):
        self.all_jobs_fld = app_config.ALL_JOBS_FOLDER

    def get_all_jobs(self):

        all_ml_jobs_listing = []

        for fld in os.listdir(self.all_jobs_fld):
            job_id_path = os.path.join(self.all_jobs_fld, fld)
            if os.path.isdir(job_id_path):  # if directory
                job_fld = os.path.join(self.all_jobs_fld, fld)
                job_created_time = datetime.fromtimestamp(os.path.getctime(job_fld)).replace(microsecond=0)
                job_last_status, job_last_status_label = self.get_job_status(fld)
                job_desc = self.get_job_desc(fld)

                job_run_status = self.get_job_run_status(fld, job_last_status)
                # job_run_status = ""

                # print("Job ID: ", fld)
                # print("Job Created Time : ", job_created_time)
                # print("Job Status: ", job_status)
                # print("Job Description : ", job_desc)

                ml_job = MLJob(fld, job_last_status_label, job_desc, job_created_time, job_run_status)

                all_ml_jobs_listing.append(ml_job)

                all_ml_jobs_listing = self.sort_jobs(all_ml_jobs_listing, "created")

        return all_ml_jobs_listing

    def sort_jobs(self, all_ml_jobs_listing, sort_by):

        if sort_by == "created":
            all_ml_jobs_listing = sorted(all_ml_jobs_listing, key=lambda x: x.job_created_time, reverse=True)

        return all_ml_jobs_listing

    def get_job_status(self, job_id):
        job_status = helper.get_job_status_detail(job_id, "status")

        if job_status is None:
            job_status_label = ""
        else:
            job_status_label = app_config.JOB_STATUS_LABELS[job_status]

        return job_status, job_status_label

    def get_job_desc(self, job_id):
        job_description = helper.get_job_status_detail(job_id, "jd_text")

        if job_description is None:
            job_description = ""

        return job_description

    def get_job_run_status(self, job_id, job_last_status):

        if job_id in running_jobs_details.keys():
            future = running_jobs_details[job_id][1]

            if future.running():
                run_status = "Running"
            elif future.cancelled():
                run_status = "Stopped"
            elif not future.exception() is None:
                run_status = "Errored"
            else:
                run_status = "Unknown"
        else:
            if job_last_status == app_config.JOB_INIT_STATUS:
                run_status = "Not Started"
            elif job_last_status == app_config.STEPS_COMPLETED_STATUS:
                run_status = "Completed"
            else:
                job_run_status = helper.get_job_status_detail(job_id, "job_run_status")

                if job_run_status is None or job_run_status.strip() == "":
                    run_status = "Unknown"
                else:
                    run_status = job_run_status

        return run_status

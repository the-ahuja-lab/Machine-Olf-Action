from email import utils

import ManageJob as manage_job
from MLJobConfig import MLJobConfig

# import ml_pipeline.utils.Logging as logging
# logger = logging.get_job_logger("/home/mohit/all_jobs/20200523015453985876/.config/logs")

class BaseMLJob:

    def __init__(self, job_id):
        print("Inside BaseMLJob initailize")
        # logger.debug("Inside BaseMLJob debug")
        # logger.info("Inside BaseMLJob initailize info")
        # logger.error("Inside BaseMLJob initailize error")
        self.job_id = job_id
        self.job_data = None
        self.config = MLJobConfig()
        self.data = None

        self.initialize()

    def initialize(self):
        config, job_data = self.get_job_data()

        ml_job_config = MLJobConfig(config)
        self.config = ml_job_config

        self.job_data = job_data

        # self.populate_data()

    # def populate_data(self):
    #     status = self.job_data['status']
    #
    #     if status is None:

    def get_job_data(self):
        job_config, job_details = manage_job.get_job_details(self.job_id)
        return job_config, job_details

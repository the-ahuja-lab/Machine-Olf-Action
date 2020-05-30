import ManageJob as manage_job
from MLJobConfig import MLJobConfig


class BaseMLJob:

    def __init__(self, job_id):
        self.job_id = job_id
        self.job_data = None
        self.config = MLJobConfig()  # initialize job with empty config
        self.data = None

        self.initialize_job()

    def initialize_job(self):
        job_user_config, job_other_details = manage_job.get_job_details(self.job_id)

        ml_job_config = MLJobConfig(job_user_config)
        self.config = ml_job_config

        self.job_data = job_other_details

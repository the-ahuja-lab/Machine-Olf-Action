import pandas as pd
import os

import MLPipeline
import AppConfig as app_config

import ml_pipeline.utils.Helper as helper


class ReadUserData:

    def __init__(self, ml_pipeline: MLPipeline):
        self.ml_pipeline = ml_pipeline
        self.jlogger = self.ml_pipeline.jlogger

        self.jlogger.info("Inside ReadUserData initialization with status {}".format(self.ml_pipeline.status))

        if self.ml_pipeline.data is None and self.ml_pipeline.status == app_config.JOB_INIT_STATUS:
            user_data_fp = os.path.join(self.ml_pipeline.job_data['job_data_path'], app_config.USER_IP_FLD_NAME,
                                        app_config.USER_IP_FNAME)
            self.read_data(user_data_fp)

    def read_data(self, fp):
        data = pd.read_csv(fp)

        if self.validate_data(data):
            self.jlogger.info("Read data is in valid format")
            self.ml_pipeline.data = data

            updated_status = app_config.STEP0_STATUS

            job_oth_config_fp = self.ml_pipeline.job_data['job_oth_config_path']
            helper.update_job_status(job_oth_config_fp, updated_status)

            self.ml_pipeline.status = updated_status
            self.jlogger.info("Read data completed successfully")

    def validate_data(self, data):
        # TODO: Validate user uploaded input, check columns format, data types, check imbalance
        return True

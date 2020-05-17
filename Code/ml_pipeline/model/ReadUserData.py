import MLPipeline
import pandas as pd
import os


class ReadUserData():

    def __init__(self, ml_pipeline: MLPipeline):
        print("Inside ReadUserData initialization")
        if ml_pipeline.data is None and ml_pipeline.status is None:
            print(ml_pipeline.job_data)
            user_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step0", "user_data.csv")
            data = self.read_data(user_data_fp)
            if data != None:
                ml_pipeline.data = data
                ml_pipeline.status = "read_data"

    def read_data(self, filepath):
        data = pd.read_csv(filepath)

        if not self.validate_data(data):
            data = None
        else:
            # TODO: Update status to 1st step completed and upload data to step 0
            pass
        return data

    def validate_data(self, data):
        # TODO: Validate user uploaded input
        return True

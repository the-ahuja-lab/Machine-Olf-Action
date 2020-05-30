from BaseMLJob import BaseMLJob
import ReadUserData as rud
import FeatureGeneration as fg
import Preprocessing as pp
import FeatureSelection as fs
import FeatureExtraction as fe
import Classification as clf
import TestSetGeneration as tsg
import TestSetPreprocessing as tspp
import TestSetPrediction as tsprd
import CrossValidation as cv

import ml_pipeline.utils.Logging as logging

logger = logging.logger


class MLPipeline(BaseMLJob):

    def __init__(self, job_id):
        super(MLPipeline, self).__init__(job_id)
        logger.debug("Initializing a new instance of MLPipeline")

        self.status = self.job_data['status']
        self.data_labels = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        # TODO double -check is this right place to initialize job level logs
        self.jlogger = logging.get_job_logger(self.job_data['job_log_path'])

    def start(self):
        # perform some logging
        self.jlogger.info("Starting job with job id {}".format(self.job_id))
        self.jlogger.debug("Job Config: {}".format(self.config))
        self.jlogger.debug("Job Other Data: {}".format(self.job_data))

        try:
            rud.ReadUserData(self)
            fg.FeatureGeneration(self, is_train=True)
            pp.Preprocessing(self, is_train=True)
            fs.FeatureSelection(self, is_train=True)
            fe.FeatureExtraction(self, is_train=True)
            clf.Classification(self)
            cv.CrossValidation(self)
            tsg.TestSetGeneration(self)
            tspp.TestSetPreprocessing(self)
            tsprd.TestSetPrediction(self)
            job_success_status = True
        except:
            job_success_status = False
            self.jlogger.exception("Exception occurred in ML Job {} ".format(self.job_id))

        return job_success_status


if __name__ == '__main__':
    ml = MLPipeline("20200415213419455528")
    ml.start()

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


class MLPipeline(BaseMLJob):

    def __init__(self, job_id):
        super(MLPipeline, self).__init__(job_id)
        print("Inside MLPipeline")

        print("Job ID: ", self.job_id)
        print("Job Data", self.job_data)
        print("Config: ", self.config)
        print("Data :", self.data)

        self.status = self.job_data['status']
        self.data_labels = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def start(self):
        rud.ReadUserData(self)
        fg.FeatureGeneration(self, is_train=True)
        pp.Preprocessing(self)
        fs.FeatureSelection(self)
        fe.FeatureExtraction(self)
        clf.Classification(self)
        cv.CrossValidation(self)
        tsg.TestSetGeneration(self)
        tspp.TestSetPreprocessing(self)
        tsprd.TestSetPrediction(self)


if __name__ == '__main__':
    ml = MLPipeline("20200415213419455528")
    ml.start()

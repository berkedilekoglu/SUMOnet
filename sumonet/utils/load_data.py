
import random


dataPath = "sumonet/data/"

def get_train_data():

        trainDataPath = dataPath + "train/"

        posTrainDataPath = trainDataPath + "Sumoylation_pos_Train.fasta"
        negTrainDataPath = trainDataPath + "Sumoylation_neg_Train.fasta"


        return posTrainDataPath, negTrainDataPath

def get_test_data():

        testDataPath = dataPath + "test/"

        posTestDataPath = testDataPath + "Sumoylation_pos_Test.fasta"
        negTestDataPath = testDataPath + "Sumoylation_neg_Test.fasta"


        return posTestDataPath, negTestDataPath



class Data:

        def __init__(self):
               
                
                self.posTrainDataPath, self.negTrainDataPath = get_train_data()
                self.posTestDataPath, self.negTestDataPath = get_test_data()


        def get_new_data(self,filePath):
        
                with open(filePath) as f:
                        
                        my_array = f.read().splitlines()
        
                return my_array[1::2]

        def all_data(self):

                posTrainData = self.get_new_data(self.posTrainDataPath)
                negTrainData = self.get_new_data(self.negTrainDataPath)

                posTestData = self.get_new_data(self.posTestDataPath)
                negTestData = self.get_new_data(self.negTestDataPath)

                return [posTrainData, negTrainData, posTestData, negTestData]

        def randomly_sample(self, data, ratio):

                return random.sample(data,k=int((ratio/2)*len(data)))


        def sample_data(self, ratio=0.4):

                
                sampledPosTrainData, sampledNegTrainData, sampledPosTestData, sampledNegTestData = list(map(self.randomly_sample,self.all_data(),[ratio]*len(self.all_data())))

                X_train = sampledPosTrainData + sampledNegTrainData
                X_test = sampledPosTestData + sampledNegTestData

                y_train = [1]*len(sampledPosTrainData) + [0]*len(sampledNegTrainData)
                y_test =  [1]*len(sampledPosTestData) + [0]*len(sampledNegTestData)


                return X_train, y_train, X_test, y_test

       
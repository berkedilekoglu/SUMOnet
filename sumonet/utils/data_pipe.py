import os

from typing import List, Tuple


script_directory = os.path.dirname(os.path.abspath(__file__))



class Data:

        def __init__(self):
               
                self.posTrainDataPath, self.negTrainDataPath = self.get_train_data_path()
                self.posTestDataPath, self.negTestDataPath = self.get_test_data_path()


        @staticmethod
        def get_train_data_path():
                """
                Function that returns train data path.
        
                Returns:
                posTrainDataPath: The data path that contains positive training samples.
                negTrainDataPath: The data path that contains negative training samples.
                
                """

                posTrainDataPath = os.path.join(script_directory, "..", "data","train","Sumoylation_pos_Train.fasta")
                negTrainDataPath = os.path.join(script_directory, "..", "data","train","Sumoylation_neg_Train.fasta") 


                return posTrainDataPath, negTrainDataPath

        @staticmethod
        def get_test_data_path():
                """
                Function that returns test data path.
        
                Returns:
                posTestDataPath: The data path that contains positive test samples.
                negTestDataPath: The data path that contains negative test samples.
                
                """

                posTestDataPath = os.path.join(script_directory, "..", "data","test","Sumoylation_pos_Test.fasta") 
                negTestDataPath = os.path.join(script_directory, "..", "data","test","Sumoylation_neg_Test.fasta")  


                return posTestDataPath, negTestDataPath

        @staticmethod
        def get_new_data(filePath:str) -> List[str]:
        
                with open(filePath) as f:
                        
                        my_array = f.read().splitlines()
        
                return my_array[1::2]

        def load_data(self) -> Tuple[List[str], List[int], List[str], List[int]]:
                """
                Function that returns our Train, Test sequences and their labels.
        
                Returns:
                X_train: Training sequences
                y_train: Training labels
                X_test: Test sequences
                y_test: Test labels
                """
                posTrainData = self.get_new_data(self.posTrainDataPath)
                negTrainData = self.get_new_data(self.negTrainDataPath)

                posTestData = self.get_new_data(self.posTestDataPath)
                negTestData = self.get_new_data(self.negTestDataPath)

                X_train = posTrainData + negTrainData
                X_test = posTestData + negTestData

                y_train = [1]*len(posTrainData) + [0]*len(negTrainData)
                y_test =  [1]*len(posTestData) + [0]*len(negTestData)


                return X_train, y_train, X_test, y_test

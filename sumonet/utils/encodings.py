import joblib
import numpy as np
import pandas as pd
import epitopepredict as ep
from pathlib import Path

modulePath = str(Path(__file__).parent.parent.resolve())

from utils.load_data import Data

def get_min_max_scaler():

        return modulePath + "/scaler/minmax_scaler.gz"



def create_dict():
    # Global function that is used for one-hot encoding
    # Function works like a map

    letterDict = {}
    letterDict["A"] = 0
    letterDict["C"] = 1
    letterDict["D"] = 2
    letterDict["E"] = 3
    letterDict["F"] = 4
    letterDict["G"] = 5
    letterDict["H"] = 6
    letterDict["I"] = 7
    letterDict["K"] = 8
    letterDict["L"] = 9
    letterDict["M"] = 10
    letterDict["N"] = 11
    letterDict["P"] = 12
    letterDict["Q"] = 13
    letterDict["R"] = 14
    letterDict["S"] = 15
    letterDict["T"] = 16
    letterDict["V"] = 17
    letterDict["W"] = 18
    letterDict["Y"] = 19
    letterDict["X"] = 20 
    return letterDict

class Encoding(Data):

    def __init__(self,encoderType,scaler=True):


        self.encoderType = encoderType
        self.scaler = scaler
        self.posData = []
        self.negData = []

        self.encodings = create_dict() # Dictionary for one-hot encoding
        self.blosum = ep.blosum62
        self.nlf = pd.read_csv('https://raw.githubusercontent.com/dmnfarrell/epitopepredict/master/epitopepredict/mhcdata/NLF.csv',index_col=0)

        self.sequences = []
        self.X = None
        self.Y = None

    def one_hot(self,data):

        oneHot_data = np.zeros((len(data),len(data[0]),len(self.encodings)))

        for sample_index, sequence in enumerate(data):
            
            for feature_index, aa in enumerate(sequence):

                encode = self.encodings[aa]
                oneHot_data[sample_index][feature_index][encode] = 1
                
        return oneHot_data     

    def blosum_helper(self,seq):
        #encode a peptide into blosum features

        x = pd.DataFrame([self.blosum[i] for i in seq]).reset_index(drop=True)

        e = x.values.flatten()    

        return e

    def bl_encoder(self,data):
        
        bl_data = list(map(self.blosum_helper,data))
            
        bl_data = np.asarray(bl_data,dtype='float32')

        return bl_data        

    def nlf_helper(self,seq):    

        x = pd.DataFrame([self.nlf[i] for i in seq]).reset_index(drop=True) 

        e = x.values.flatten()

        return e
        
    def nlf_encoder(self,data):
                
        nlf_data = list(map(self.nlf_helper,data))

        nlf_data = np.asarray(nlf_data,dtype='float32')

        return nlf_data     

    def labels(self):

        return np.asarray([1]*len(self.posData) + [0]*len(self.negData))

    def encode(self):

        if self.encoderType.lower() == 'blosum62':
            
            return self.bl_encoder(self.sequences)

        elif self.encoderType.lower() == 'nlf':
            
            return self.nlf_encoder(self.sequences)

        elif self.encoderType.lower() == 'one-hot':
            
            return self.one_hot(self.sequences)

    def get_encoded_vectors_from_path(self,posDataPath,negDataPath):

        """
        This function is used to get encoded vectors.

        Args:

            posDataPath: positive data path. This data file is in .fasta format.
            negDataPath: negative data path. This data file is in .fasta format.

        Output:

            self.X: Encoded samples
            self.Y: Labels
        """

        self.posData = super().get_new_data(posDataPath)
        self.negData = super().get_new_data(negDataPath)
        self.sequences = self.posData + self.negData

        self.X = self.encode()
        self.Y = self.labels()

        return self.X, self.Y


    def get_encoded_vectors_from_data(self,X):

        """
        This function is used to get encoded representation of given vectors.

        Args:

            X: List that contains samples.

        Output:

            self.X: Encoded samples
        """

        self.sequences = X

        self.X = self.encode()

        return self.X

    def minmax(self, X):

        minmax_scaler = joblib.load(get_min_max_scaler())
        return minmax_scaler.transform(X)

    def reshape(self,X):

        return X.reshape(len(X),21,X.shape[1]//21)
    
    

import joblib
import numpy as np
import pandas as pd
import epitopepredict as ep

from sumonet.utils.load_data import Data

def get_min_max_scaler_path():

        return "sumonet/utils/scaler/minmax_scaler.gz"



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

    def __init__(self,encoderType='blosum62',scaler=True):

        super().__init__()

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

    def set_encoder_type(self,encoderType):
        self.encoderType = encoderType

    def one_hot(self,data):

        oneHot_data = np.zeros((len(data),len(data[0]),len(self.encodings)))

        for sample_index, sequence in enumerate(data):
            
            for feature_index, aa in enumerate(sequence):

                encode = self.encodings[aa]
                oneHot_data[sample_index][feature_index][encode] = 1
                
        self.X = oneHot_data     

    def blosum_helper(self,seq):
        #encode a peptide into blosum features

        x = pd.DataFrame([self.blosum[i] for i in seq]).reset_index(drop=True)

        e = x.values.flatten()    

        return e

    def bl_encoder(self,data):
        
        bl_data = list(map(self.blosum_helper,data))
            
        bl_data = np.asarray(bl_data,dtype='float32')

        self.X = bl_data        

    def nlf_helper(self,seq):    

        x = pd.DataFrame([self.nlf[i] for i in seq]).reset_index(drop=True) 

        e = x.values.flatten()

        return e
        
    def nlf_encoder(self,data):
                
        nlf_data = list(map(self.nlf_helper,data))

        nlf_data = np.asarray(nlf_data,dtype='float32')

        self.X = nlf_data     

    def labels(self):

        self.Y = np.asarray([1]*len(self.posData) + [0]*len(self.negData))

    def encode(self):

        if self.encoderType.lower() == 'blosum62':
            
            self.bl_encoder(self.sequences)

        elif self.encoderType.lower() == 'nlf':
            
            self.nlf_encoder(self.sequences)

        elif self.encoderType.lower() == 'one-hot':
            
            self.one_hot(self.sequences)

        else:
            raise ValueError('EncoderType is {encoderType}. It should be blosum62, nlf or one-hot.')

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

        self.encode()
        self.labels()

        self.preprocess()
        self.preprocess_labels()

        return self.X, self.Y


    def get_encoded_vectors_from_data(self,X, Y):

        """
        This function is used to get encoded representation of given vectors.

        Args:

            X: List that contains samples.

        Output:

            self.X: Encoded samples
        """

        self.sequences = X
        self.Y = np.asarray(Y)

        self.encode()

        self.preprocess()
        self.preprocess_labels()

        return self.X, self.Y
    
    def minmax(self):

        minmax_scaler = joblib.load(get_min_max_scaler_path())
        self.X = minmax_scaler.transform(self.X)

    def reshape(self):

        self.X = self.X.reshape(len(self.X),21,self.X.shape[1]//21)

    def preprocess(self):
        
        if self.encoderType.lower() == 'blosum62' or self.encoderType.lower() == 'nlf':
            
            if self.scaler:
                self.minmax()

            self.reshape()

    def preprocess_labels(self):

            self.Y = np.eye(2)[self.Y]


    
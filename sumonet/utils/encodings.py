import numpy as np
import pandas as pd
import epitopepredict as ep

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

class Encoding:

    def __init__(self,encoderType):


        self.encoderType = encoderType

        self.posData = []
        self.negData = []

        self.encodings = create_dict() # Dictionary for one-hot encoding
        self.blosum = ep.blosum62
        self.nlf = pd.read_csv('https://raw.githubusercontent.com/dmnfarrell/epitopepredict/master/epitopepredict/mhcdata/NLF.csv',index_col=0)

        self.sequences = []
        self.X = None
        self.Y = None

    def get_new_data(self,filePath):
    
        with open(filePath) as f:
                
                my_array = f.read().splitlines()
    
        return my_array[1::2] 

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
            
            return self.bl_encoder(self.sequences)

    def get_encoded_vectors(self,posDataPath,negDataPath):

        """
        This function is used to get encoded vectors.

        Args:

            posDataPath: positive data path. This data file is in .fasta format.
            negDataPath: negative data path. This data file is in .fasta format.

        Output:

            self.X: Encoded samples
            self.Y: Labels
        """

        self.posData = self.get_new_data(posDataPath)
        self.negData = self.get_new_data(negDataPath)
        self.sequences = self.posData + self.negData

        self.X = self.encode()
        self.Y = self.labels()

        return self.X, self.Y


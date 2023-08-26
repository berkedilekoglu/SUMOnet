import os
import requests as r

from Bio import SeqIO
from io import StringIO
from typing import List, Tuple
from os.path import dirname, abspath


parent_dir = dirname(dirname(abspath(__file__))) 



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

                posTrainDataPath =  os.path.join(parent_dir,"data","train","Sumoylation_pos_Train.fasta")
                negTrainDataPath = os.path.join(parent_dir,"data","train","Sumoylation_neg_Train.fasta") 


                return posTrainDataPath, negTrainDataPath

        @staticmethod
        def get_test_data_path():
                """
                Function that returns test data path.
        
                Returns:
                posTestDataPath: The data path that contains positive test samples.
                negTestDataPath: The data path that contains negative test samples.
                
                """

                posTestDataPath = os.path.join(parent_dir,"data","test","Sumoylation_pos_Test.fasta") 
                negTestDataPath = os.path.join(parent_dir,"data","test","Sumoylation_neg_Test.fasta")  


                return posTestDataPath, negTestDataPath

        @staticmethod
        def get_new_data(filePath:str) -> List[str]:
        
                with open(filePath) as f:
                        
                        my_array = f.read().splitlines()
        
                return my_array[1::2]
        
        @staticmethod
        def retrive_protein_sequence_with_uniprotid(uniprot_id:str) -> str :

                """
                Fetches the protein sequence for a given UniProt ID.
                
                Parameters:
                        uniprot_id (str): The UniProt ID of the protein.
                        
                Returns:
                        str: The protein sequence if successful, None otherwise.
                """

                baseUrl="http://www.uniprot.org/uniprot/"
                currentUrl=baseUrl+uniprot_id+".fasta"

                try:
                        response = r.post(currentUrl)
                        cData=''.join(response.text)

                        Seq=StringIO(cData)

                        protein_sequence = list(SeqIO.parse(Seq,'fasta'))[0].seq

                        return str(protein_sequence)
                
                except:

                        return None
                
        @staticmethod
        def extract_subseq_with_k_position(sequence:str,position:int) -> str:

                """
                Take protein sequence and 'K' position as an input and find 21-mer which includes 'K' at the middle. Use padding with 'X'.
                
                Parameters:
                        sequence (str): Amino acid sequence of the protein.
                        position (int): Position of the 'K'. It is not array index. Position is index + 1.
                        
                Returns:
                        str: 21-mer which includes 'K' at the middle
                """

                half_mer_len = 10

                i = position - 1

                if sequence[i] == 'K':

                        left_side = sequence[max(0,i-half_mer_len):i]
                        right_side = sequence[i+1:i+1+half_mer_len]

                        if len(left_side) < 10:

                                left_side = 'X' * (10-len(left_side)) + left_side

                        if len(right_side) < 10:

                                right_side = right_side + 'X' * (10-len(right_side))


                        subseq = left_side + 'K' + right_side

                        return subseq
                
                else:
                        #Todo:
                        #Giving an error might be better
                        #User should be noticed for that situation!
                        return None
        
        def find_mers_with_K(self,sequence:str) -> List[str]: 

                """
                Take protein sequence as an input and find each 21-mer which includes 'K' at the middle. Use padding with 'X'.
                
                Parameters:
                        sequence (str): Amino acid sequence of the protein.
                        
                Returns:
                        list: List of 21-mers which includes 'K' at the middle
                """

                mers = []
                k_positions = []
                for i in range(len(sequence)):

                        if sequence[i] == 'K':
                                position = i+1
                                subseq = self.extract_subseq_with_k_position(sequence,position)

                                
                                mers.append(subseq)
                                k_positions.append(position)

                return mers, k_positions


        

        def protein_sequence_input(self,sequence_fasta_list:List[str]) -> Tuple[List[str], List[str]]:

                protein_ids, protein_seqs,k_positions = [], [], []

                for index, item in enumerate(sequence_fasta_list):

                        if index % 2 == 0:

                                protein_id = item

                        else:

                                mers, k_position = self.find_mers_with_K(item)
                                
                                protein_seqs += mers
                                k_positions += k_position
                                protein_ids += [protein_id] * len(mers)
                        

                return protein_ids, protein_seqs, k_positions

        def uniprot_id_input(self,protein_sequence,uniprot_id,lysine_position=None):

                protein_ids, protein_seqs, k_positions = [], [], []

                if lysine_position == None:

                        protein_seqs, k_positions = self.find_mers_with_K(protein_sequence)
                        protein_ids+=[uniprot_id] * len(protein_seqs)

                else:

                        subseq = self.extract_subseq_with_k_position(protein_sequence,lysine_position)   
                        
                        protein_ids.append(uniprot_id)
                        protein_seqs.append(subseq)
                        k_positions.append(lysine_position)

                return protein_ids, protein_seqs, k_positions

        def fasta_file_input(self,readed_file):

                content_list = readed_file.read().decode("utf-8").splitlines()        

                return self.protein_sequence_input(content_list)

        def load_sumonet_experiment_data(self) -> Tuple[List[str], List[int], List[str], List[int]]:
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

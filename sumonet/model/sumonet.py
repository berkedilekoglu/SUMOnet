 
from tensorflow.keras.models import load_model

from model.architecture import Architecture
from pathlib import Path


modulePath = str(Path(__file__).parent.parent.resolve())

def get_model_path():

    modelPath = modulePath + '/model/pretrained/'

    return modelPath + 'sumonet3.h5'


class SUMOnet(Architecture):

        def __init__(self):

            super(SUMOnet,self).__init__()
                
            self.model = Architecture()

        def set_weights(self):

            preTrainedModelPath = get_model_path()
            self.model = load_model(preTrainedModelPath)
        
       

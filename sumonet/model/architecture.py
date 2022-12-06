
from tensorflow.keras import layers, Model, regularizers


def get_model_path(state='final'):


    modelPath = 'sumonet/model/pretrained/'

    if state == 'final':

    	print('This model was trained on entire (Train + Test) data! If you want to use model that was trained on only Train samples please use load_weights(model_state=\'partial\')')
    	return modelPath + 'sumonet3.h5'

    elif state == 'partial':
    	print('This model was trained on the Train data! If you want to use final model please use load_weights(model_state=\'final\') ')
    	return modelPath + 'sumonet3_partial.h5'

    else:

    	raise ValueError('model_state just takes \'final\' or \'partial\' parameters')


class SUMOnet(Model):

    def __init__(self):
        
        super().__init__()
        
        self.cnn = layers.Conv1D(128,2,padding='valid',activation='relu',kernel_initializer='he_normal',strides=1)
        self.bigru = layers.Bidirectional(layers.GRU(16, dropout=0.4, recurrent_dropout=0,return_sequences=True))
        self.pool = layers.GlobalAveragePooling1D()
        self.dense64 = layers.Dense(64,kernel_initializer='he_normal',activity_regularizer= regularizers.l2(1e-4))
        self.dropout = layers.Dropout(0.4)
        self.relu = layers.Activation('relu')
        self.dense128_1 = layers.Dense(128,kernel_initializer='he_normal',activity_regularizer= regularizers.l2(1e-4))
        self.dropout_1 = layers.Dropout(0.4)
        self.relu_1 = layers.Activation('relu')
        self.dense128_2 = layers.Dense(128,kernel_initializer='he_normal',activity_regularizer= regularizers.l2(1e-4))
        self.dropout_2 = layers.Dropout(0.4)
        self.relu_2 = layers.Activation('relu')
        self.dense2 = layers.Dense(2, kernel_initializer='he_normal')
        self.softmax = layers.Activation('softmax')


    def call(self, inputs):

        x = self.cnn(inputs)
        x = self.bigru(x)
        x = self.pool(x)
        x = self.dense64(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dense128_1(x)
        x = self.dropout_1(x)
        x = self.relu_1(x)
        x = self.dense128_2(x)
        x = self.dropout_2(x)
        x = self.relu_2(x)
        x = self.dense2(x)
        x = self.softmax(x)
        
        return x

    def load_weights(self,model_state='final'):

        preTrainedModelPath = get_model_path(model_state)
        super().load_weights(preTrainedModelPath)


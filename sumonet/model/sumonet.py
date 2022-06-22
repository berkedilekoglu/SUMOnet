from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

class SUMOnet(layers.Layer):

    def __init__(self):
        
        super(SUMOnet,self).__init__()

        self.cnn = layers.Conv1D(128,2,padding='valid',activation='relu',kernel_initializer='he_normal',strides=1)
        self.bigru = layers.Bidirectional(layers.GRU(32, dropout=0.4, recurrent_dropout=0,return_sequences=True))
        self.pool = layers.GlobalAveragePooling1D()
        self.dense64 = layers.Dense(64,kernel_initializer='he_normal',activity_regularizer= l2(1e-4))
        self.dropout = layers.Dropout(0.4)
        self.relu = layers.Activation('relu')
        self.dense128 = layers.Dense(128,kernel_initializer='he_normal',activity_regularizer= l2(1e-4))
        self.dense2 = layers.Dense(2, kernel_initializer='he_normal')
        self.softmax = layers.Activation('softmax')




    def call(self, inputs):

        x = self.cnn(inputs)
        x = self.bigru(x)
        x = self.pool(x)
        x = self.dense64(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dense128(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dense128(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.softmax(x)

        return x

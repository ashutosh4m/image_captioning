import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tensorflow.keras.layers import Embedding, LSTM, Dense, Add, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3

def define_model(vocab_size, max_length):
    image_model = InceptionV3(include_top=False, pooling='avg')
    image_model.trainable = False
    
    inputs1 = Input(shape=(2048,))
    fe1 = Dense(256, activation='relu')(inputs1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)
    
    decoder1 = Add()([fe1, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

if __name__ == "__main__":
    vocab_size = 10000
    max_length = 34
    model = define_model(vocab_size, max_length)
    model.summary()

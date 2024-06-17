import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('models/model.keras', monitor='loss', verbose=1, save_best_only=True, mode='min')





from src.models.model import define_model
from src.utils.utils import data_generator
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

def train_model(model, train_generator, validation_generator, steps_per_epoch, validation_steps, epochs=10):
    checkpoint = ModelCheckpoint('models/model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint]
    )
    return history

if __name__ == "__main__":
    vocab_size = 10000
    max_length = 34
    model = define_model(vocab_size, max_length)
    
    descriptions = {'img1': ['a dog is running'], 'img2': ['a cat is sleeping']}
    photos = {'img1': np.random.rand(2048), 'img2': np.random.rand(2048)}
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([desc for desc_list in descriptions.values() for desc in desc_list])
    batch_size = 2
    steps_per_epoch = 1
    validation_steps = 1

    train_generator = data_generator(descriptions, photos, tokenizer, max_length, vocab_size, batch_size)
    validation_generator = data_generator(descriptions, photos, tokenizer, max_length, vocab_size, batch_size)

    history = train_model(model, train_generator, validation_generator, steps_per_epoch, validation_steps, epochs=1)
    print("Training completed")

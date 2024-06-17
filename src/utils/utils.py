import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def load_captions(captions_file):
    with open(captions_file, 'r') as file:
        captions = file.readlines()
    return captions

def load_images(image_dir):
    image_files = os.listdir(image_dir)
    return image_files

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((299, 299))
    image = np.array(image) / 255.0
    return image

def load_and_preprocess_images(image_dir, image_files):
    images = {}
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        if os.path.exists(image_path):
            images[image_file] = preprocess_image(image_path)
        else:
            print(f"Warning: {image_path} not found, skipping.")
    return images

def split_data(image_ids, test_size=0.2, random_state=42):
    train_ids, test_ids = train_test_split(image_ids, test_size=test_size, random_state=random_state)
    return train_ids, test_ids

def clean_caption(caption):
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    return caption

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size, batch_size):
    while True:
        X1, X2, y = [], [], []
        for key, desc_list in descriptions.items():
            if key not in photos:  # Skip keys with missing images
                continue
            photo = photos[key]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
                    if len(X1) == batch_size:
                        yield ([np.array(X1), np.array(X2)], np.array(y))
                        X1, X2, y = [], [], []
                    elif len(X1) > batch_size:
                        yield ([np.array(X1[:batch_size]), np.array(X2[:batch_size])], np.array(y[:batch_size]))
                        X1, X2, y = X1[batch_size:], X2[batch_size:], y[batch_size:]





if __name__ == "__main__":
    captions_file = 'data/raw/Flickr8k_text/Flickr8k.token.txt'
    captions = load_captions(captions_file)
    print(f"Loaded {len(captions)} captions")

    image_dir = 'data/raw/Flickr8k_Dataset/Flicker8k_Dataset'
    image_files = load_images(image_dir)
    print(f"Loaded {len(image_files)} images")

    if image_files:
        sample_image_path = os.path.join(image_dir, image_files[0])
        image = preprocess_image(sample_image_path)
        print(f"Sample image shape: {image.shape}")

    images = load_and_preprocess_images(image_dir, image_files[:5])
    print(f"Preprocessed {len(images)} images")

    train_ids, test_ids = split_data(image_files)
    print(f"Train/Test split: {len(train_ids)}/{len(test_ids)}")

    sample_caption = "A man is riding a bike."
    cleaned_caption = clean_caption(sample_caption)
    print(f"Cleaned caption: {cleaned_caption}")

    descriptions = {'img1': ['a dog is running'], 'img2': ['a cat is sleeping']}
    photos = {'img1': np.random.rand(2048), 'img2': np.random.rand(2048)}
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([desc for desc_list in descriptions.values() for desc in desc_list])
    max_length = 5
    vocab_size = len(tokenizer.word_index) + 1
    batch_size = 2

    generator = data_generator(descriptions, photos, tokenizer, max_length, vocab_size, batch_size)
    while True:
        try:
            X, y = next(generator)
            print(f"Generated batch shapes: X[0]={X[0].shape}, X[1]={X[1].shape}, y={y.shape}")
        except StopIteration:
            break
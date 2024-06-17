import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.preprocess import preprocess_image, load_images, split_data
from models.model import define_model
from training.train import train_model
from utils.utils import load_captions, clean_caption, data_generator
from evaluation.evaluate import plot_training_history, evaluate_model, calculate_bleu_scores
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

if __name__ == "__main__":
    # Load data
    captions_file = 'data/raw/Flickr8k_text/Flickr8k.token.txt'
    image_dir = 'data/raw/Flickr8k_Dataset/Flicker8k_Dataset'
    captions = load_captions(captions_file)
    image_files = load_images(image_dir)

    # Extract and clean captions
    caption_dict = {image_file.split('.')[0]: [clean_caption(caption.strip().split('\t')[1]) for caption in captions if caption.startswith(image_file.split('.')[0])] for image_file in image_files}
    cleaned_captions = {k: [clean_caption(c) for c in v] for k, v in caption_dict.items()}

    # Tokenize captions
    all_captions = [caption for captions in cleaned_captions.values() for caption in captions]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for c in all_captions)

    # Save tokenizer for later use
    with open('tokenizer.pkl', 'wb') as file:
        pickle.dump(tokenizer, file)

    # Load and preprocess images
    images = load_and_preprocess_images(image_dir, image_files)

    # Split data
    train_ids, test_ids = split_data(list(images.keys()))

    # Define model
    model = define_model(vocab_size, max_length)
    model.summary()

    # Prepare the data generators
    train_generator = data_generator({k: cleaned_captions[k] for k in train_ids}, {k: images[k] for k in train_ids}, tokenizer, max_length, vocab_size, batch_size=32)
    validation_generator = data_generator({k: cleaned_captions[k] for k in test_ids}, {k: images[k] for k in test_ids}, tokenizer, max_length, vocab_size, batch_size=32)

    # Train model
    steps_per_epoch = len(train_ids) // 32
    validation_steps = len(test_ids) // 32
    history = train_model(model, train_generator, validation_generator, steps_per_epoch, validation_steps, epochs=10)

    # Evaluate model
    plot_training_history(history)

    # Evaluate the model on the test set
    actual, predicted = evaluate_model(model, {k: cleaned_captions[k] for k in test_ids}, {k: images[k] for k in test_ids}, tokenizer, max_length)

    # Calculate BLEU scores
    bleu1, bleu2, bleu3, bleu4 = calculate_bleu_scores(actual, predicted)
    print(f'BLEU-1: {bleu1}')
    print(f'BLEU-2: {bleu2}')
    print(f'BLEU-3: {bleu3}')
    print(f'BLEU-4: {bleu4}')

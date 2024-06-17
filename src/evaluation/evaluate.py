import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
from src.models.model import define_model

def plot_training_history(history):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    for key, desc_list in descriptions.items():
        yhat = generate_caption(model, tokenizer, photos[key], max_length)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    return actual, predicted

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def calculate_bleu_scores(actual, predicted):
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1, bleu2, bleu3, bleu4

if __name__ == "__main__":
    # Dummy history object
    class DummyHistory:
        history = {'loss': [1, 0.9], 'val_loss': [1.1, 1]}
    
    history = DummyHistory()
    plot_training_history(history)
    
    # Dummy data for testing
    vocab_size = 10000
    max_length = 34
    model = define_model(vocab_size, max_length)
    descriptions = {'img1': ['a dog is running'], 'img2': ['a cat is sleeping']}
    photos = {'img1': np.random.rand(1, 2048), 'img2': np.random.rand(1, 2048)}
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([desc for desc_list in descriptions.values() for desc in desc_list])

    actual, predicted = evaluate_model(model, descriptions, photos, tokenizer, max_length)
    bleu1, bleu2, bleu3, bleu4 = calculate_bleu_scores(actual, predicted)
    
    print(f'BLEU-1: {bleu1}')
    print(f'BLEU-2: {bleu2}')
    print(f'BLEU-3: {bleu3}')
    print(f'BLEU-4: {bleu4}')
    
    print("Evaluation functions test passed.")

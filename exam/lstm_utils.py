# # exam/lstm_utils.py
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense, Activation
# from tensorflow.keras.optimizers import RMSprop
# import pickle
# import os
# import random

# class TextGenerator:
#     def __init__(self, model_dir='ml_models'):
#         self.model_dir = model_dir
#         if not os.path.exists(self.model_dir):
#             os.makedirs(self.model_dir)
            
#         self.model_path = os.path.join(self.model_dir, 'lstm_model.keras')
#         self.tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pkl')
#         self.seq_length = 40

#     def prepare_data(self, text_path):
#         # Load text
#         with open(text_path, 'r', encoding='utf-8') as f:
#             text = f.read().lower()
            
#         print(f'Corpus length: {len(text)}')

#         # Create mapping of unique characters to integers
#         chars = sorted(list(set(text)))
#         char_indices = dict((c, i) for i, c in enumerate(chars))
#         indices_char = dict((i, c) for i, c in enumerate(chars))

#         # Build sequences
#         step = 3
#         sentences = []
#         next_chars = []
#         for i in range(0, len(text) - self.seq_length, step):
#             sentences.append(text[i : i + self.seq_length])
#             next_chars.append(text[i + self.seq_length])

#         # Vectorize (One-hot encoding)
#         x = np.zeros((len(sentences), self.seq_length, len(chars)), dtype=bool)
#         y = np.zeros((len(sentences), len(chars)), dtype=bool)
        
#         for i, sentence in enumerate(sentences):
#             for t, char in enumerate(sentence):
#                 x[i, t, char_indices[char]] = 1
#             y[i, char_indices[next_chars[i]]] = 1
            
#         return x, y, chars, char_indices, indices_char

#     def build_and_train(self, text_path, epochs=5):
#         x, y, chars, char_indices, indices_char = self.prepare_data(text_path)
        
#         # Save mappings for later use
#         with open(self.tokenizer_path, 'wb') as f:
#             pickle.dump({'chars': chars, 'char_indices': char_indices, 'indices_char': indices_char}, f)

#         # Build LSTM Model
#         model = Sequential()
#         model.add(LSTM(128, input_shape=(self.seq_length, len(chars))))
#         model.add(Dense(len(chars)))
#         model.add(Activation('softmax'))

#         optimizer = RMSprop(learning_rate=0.01)
#         model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#         print("Training model...")
#         model.fit(x, y, batch_size=128, epochs=epochs)
#         model.save(self.model_path)
#         print("Model saved.")

#     def generate_text(self, seed_text, length=400):
#         if not os.path.exists(self.model_path):
#             return "Error: Model not trained yet. Please run the training command."

#         # Load model and mappings
#         model = load_model(self.model_path)
#         with open(self.tokenizer_path, 'rb') as f:
#             data = pickle.load(f)
#             chars = data['chars']
#             char_indices = data['char_indices']
#             indices_char = data['indices_char']

#         # Normalize seed
#         generated = ''
#         sentence = seed_text.lower()
        
#         # Pad seed if it's too short
#         if len(sentence) < self.seq_length:
#             sentence = sentence.rjust(self.seq_length, ' ')
        
#         # Take only the last 'seq_length' characters
#         sentence = sentence[-self.seq_length:]
        
#         generated += sentence

#         result_text = seed_text # Keep original casing for display if desired

#         for i in range(length):
#             x_pred = np.zeros((1, self.seq_length, len(chars)))
#             for t, char in enumerate(sentence):
#                 if char in char_indices:
#                     x_pred[0, t, char_indices[char]] = 1.

#             preds = model.predict(x_pred, verbose=0)[0]
#             next_index = self.sample(preds, temperature=0.5) # Adjust temperature for creativity
#             next_char = indices_char[next_index]

#             sentence = sentence[1:] + next_char
#             result_text += next_char

#         return result_text

#     def sample(self, preds, temperature=1.0):
#         # Helper function to sample an index from a probability array
#         preds = np.asarray(preds).astype('float64')
#         preds = np.log(preds + 1e-7) / temperature
#         exp_preds = np.exp(preds)
#         preds = exp_preds / np.sum(exp_preds)
#         probas = np.random.multinomial(1, preds, 1)
#         return np.argmax(probas)



import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
import string

class TextGenerator:
    def __init__(self, model_dir='ml_models'):
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.model_path = os.path.join(self.model_dir, 'lstm_model.keras')
        self.tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pkl')
        self.seq_length = 40

    def prepare_data(self, text_path):
        # 1. Load text
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 2. Preprocessing: Lowercase and Remove Punctuation 
        text = text.lower()
        # Create a translation table to remove all punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
            
        print(f'Corpus length (cleaned): {len(text)}')

        # Create mapping of unique characters to integers
        chars = sorted(list(set(text)))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        # Build sequences
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(text) - self.seq_length, step):
            sentences.append(text[i : i + self.seq_length])
            next_chars.append(text[i + self.seq_length])

        # 3. Vectorize for Embedding Layer
        # Input 'x' is now integers, not one-hot vectors
        x = np.zeros((len(sentences), self.seq_length), dtype=np.int32)
        y = np.zeros((len(sentences), len(chars)), dtype=bool) # Output remains one-hot for Softmax
        
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t] = char_indices[char]
            y[i, char_indices[next_chars[i]]] = 1
            
        return x, y, chars, char_indices, indices_char

    def build_and_train(self, text_path, epochs=10):
        x, y, chars, char_indices, indices_char = self.prepare_data(text_path)
        
        vocab_size = len(chars)
        
        # Save mappings
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump({'chars': chars, 'char_indices': char_indices, 'indices_char': indices_char}, f)

        # 4. Model Design with Embedding Layer 
        model = Sequential()
        # Embedding layer: Maps integer indices to dense vectors
        model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=self.seq_length))
        model.add(LSTM(128)) # Removed input_shape, handled by Embedding
        model.add(Dense(vocab_size))
        model.add(Activation('softmax'))

        # Compile with Adam optimizer 
        optimizer = Adam(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # 5. Training with Early Stopping and Validation Split [cite: 27, 29]
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        print("Training model...")
        model.fit(x, y, 
                  batch_size=128, 
                  epochs=epochs, 
                  validation_split=0.2,  # Split 20% for validation
                  callbacks=[early_stopping]) # Add early stopping
        
        model.save(self.model_path)
        print("Model saved.")

    def generate_text(self, seed_text, length=400):
        if not os.path.exists(self.model_path):
            return "Error: Model not trained yet."

        model = load_model(self.model_path)
        with open(self.tokenizer_path, 'rb') as f:
            data = pickle.load(f)
            chars = data['chars']
            char_indices = data['char_indices']
            indices_char = data['indices_char']

        # Preprocess seed similarly (remove punctuation)
        sentence = seed_text.lower().translate(str.maketrans('', '', string.punctuation))
        
        if len(sentence) < self.seq_length:
            sentence = sentence.rjust(self.seq_length, ' ')
        
        sentence = sentence[-self.seq_length:]
        result_text = seed_text # Display original input to user, but model sees cleaned version

        for i in range(length):
            # Prepare input for prediction (Integer sequence for Embedding)
            x_pred = np.zeros((1, self.seq_length))
            for t, char in enumerate(sentence):
                if char in char_indices:
                    x_pred[0, t] = char_indices[char]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, temperature=0.5)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char
            result_text += next_char

        return result_text

    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-7) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
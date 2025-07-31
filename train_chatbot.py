# In train_chatbot.py

import pandas as pd
import numpy as np
import os
import pickle
import time
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# --- Constants ---
EMBEDDING_DIM = 100
LSTM_UNITS = 128
BATCH_SIZE = 32
EPOCHS = 10
VOCAB_SIZE_LIMIT = 20000 # Increased slightly for more vocabulary

# --- 1. Load NEW Data ---
print("Loading FINAL (email-friendly) processed data...")
# --- CHANGE: Load the new CSV file ---
df = pd.read_csv('data/customer_support_pairs_final.csv') 
df.dropna(inplace=True)
df = df.sample(n=25000, random_state=42)
print(f"Data shape: {df.shape}")

# --- 2. Pre-process for Seq2Seq ---
df['response'] = df['response'].apply(lambda x: '<start> ' + str(x) + ' <end>')

# --- 3. Tokenization ---
print("Tokenizing text with email-friendly filters...")
# --- CHANGE: Update filters to allow @ and . and <> ---
filters = '!"#$%&()*+,-/:;=?[\\]^_`{|}~\t\n'
tokenizer = Tokenizer(num_words=VOCAB_SIZE_LIMIT, oov_token='<unk>', filters=filters)

all_texts = list(df['input']) + list(df['response'])
tokenizer.fit_on_texts(all_texts)
word_index = tokenizer.word_index

# --- CHANGE: Save with a new name ---
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
with open('saved_models/tokenizer_final.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Final tokenizer saved.")

# ... (The rest of the script is the same as the 'Lighter Version' you ran before) ...

# --- 4. Convert to Sequences ---
input_sequences = tokenizer.texts_to_sequences(df['input'])
response_sequences = tokenizer.texts_to_sequences(df['response'])
vocab_size = min(len(word_index) + 1, VOCAB_SIZE_LIMIT)
print(f"Vocabulary Size (limited): {vocab_size}")

# --- 5. Padding ---
input_padded = pad_sequences(input_sequences, padding='post')
response_padded = pad_sequences(response_sequences, padding='post')
max_input_length = input_padded.shape[1]
max_response_length = response_padded.shape[1]

# --- 6. Word2Vec ---
print("\nTraining Word2Vec model...")
sentences = [text.split() for text in all_texts]
w2v_model = Word2Vec(sentences=sentences, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= VOCAB_SIZE_LIMIT: continue
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
print("Embedding matrix created.")

# --- 7. Create Datasets ---
encoder_input_data = input_padded
decoder_input_data = response_padded[:, :-1]
decoder_target_data = response_padded[:, 1:]
X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
    encoder_input_data, decoder_input_data, decoder_target_data, test_size=0.1, random_state=42
)
print("Data preparation complete!")

# --- 8. Build & Compile Model ---
print("\nBuilding the model...")
encoder_inputs = Input(shape=(max_input_length,))
emb_layer = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)
encoder_embedding = emb_layer(encoder_inputs)
encoder_lstm = LSTM(LSTM_UNITS, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(max_response_length - 1,))
decoder_embedding = emb_layer(decoder_inputs)
decoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 9. Train ---
print("\nStarting FINAL model training...")
model.fit(
    [X_train, y1_train], y2_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=([X_val, y1_val], y2_val)
)

# --- 10. Save ---
# --- CHANGE: Save with a new name ---
model.save('saved_models/seq2seq_model_final.keras')
print("Final trained model saved successfully to 'saved_models/seq2seq_model_final.keras'")
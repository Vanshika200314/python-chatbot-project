# In chat.py (with Order and Flight Cancellation Logic)

import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import random
import re

# --- (Loading and model rebuilding is the same as before) ---
print("Loading FINAL trained model and tokenizer...")
model_path = os.path.join('saved_models', 'seq2seq_model_final.keras')
tokenizer_path = os.path.join('saved_models', 'tokenizer_final.pickle')
training_model = load_model(model_path)
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Assets loaded successfully.")
max_input_length = training_model.input_shape[0][1]
max_response_length = training_model.input_shape[1][1] + 1
lstm_units = training_model.layers[4].units
encoder_inputs = training_model.input[0]
_, state_h_enc, state_c_enc = training_model.layers[3].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)
decoder_inputs = training_model.input[1]
decoder_state_input_h = Input(shape=(lstm_units,), name='input_3')
decoder_state_input_c = Input(shape=(lstm_units,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
emb_layer = training_model.layers[2]
decoder_embedding = emb_layer(decoder_inputs)
decoder_lstm = training_model.layers[4]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = training_model.layers[5]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)
print("Inference models built.")
reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

def decode_sequence_beam(input_seq, beam_width=3):
    states_value = encoder_model.predict(input_seq, verbose=0)
    start_token_index = tokenizer.word_index['<start>']
    beams = [([start_token_index], 0.0, states_value)]
    for _ in range(max_response_length):
        all_candidates = []
        for seq, score, states in beams:
            if seq[-1] == tokenizer.word_index.get('<end>'):
                all_candidates.append((seq, score, states))
                continue
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = seq[-1]
            output_tokens, h, c = decoder_model.predict([target_seq] + states, verbose=0)
            word_probabilities = output_tokens[0, -1, :]
            top_next_indices = np.argsort(word_probabilities)[-beam_width:]
            for word_index in top_next_indices:
                new_seq = seq + [word_index]
                new_score = score - np.log(word_probabilities[word_index] + 1e-9)
                all_candidates.append((new_seq, new_score, [h, c]))
        beams = sorted(all_candidates, key=lambda x: x[1])[:beam_width]
        if all(b[0][-1] == tokenizer.word_index.get('<end>') for b in beams):
            break
    best_seq = beams[0][0]
    decoded_sentence = ''
    for token_index in best_seq:
        word = reverse_word_index.get(token_index)
        if word and word not in ['<start>', '<end>', '<unk>']:
            decoded_sentence += ' ' + word
    return decoded_sentence.strip()


# --- THE DEFINITIVE CONVERSATION MANAGER (with Flight Logic) ---
def get_final_response(user_input, conversation_state):
    # Rule 1: Handle Farewells
    if any(word in user_input.lower() for word in ['bye', 'goodbye', 'see you', 'farewell', 'quit']):
        return "Thank you for chatting with me. Goodbye!", "general"

    # Rule 2: Handle awaiting_email state
    elif conversation_state == "awaiting_email":
        if re.search(r'\S+@\S+', user_input):
            return "Thank you. I have your contact information now. How can I assist you further?", "has_email"
        else:
            return "I'll need a valid email address to proceed. Could you please provide it?", "awaiting_email"
    
    # Rule 3: Handle awaiting_order_number state
    elif conversation_state == "awaiting_order_number":
        match = re.search(r'\d{5,}', user_input)
        if match:
            order_number = match.group(0)
            response_text = f"Thank you. I've looked up order number {order_number} and it is currently out for delivery."
            return response_text, "has_email"
        else:
            response_text = "I'm sorry, that doesn't look like a valid order number. It should be at least 5 digits long. Could you please provide it?"
            return response_text, "awaiting_order_number"

    # *** NEW SKILL - START ***
    # Rule 4: Handle awaiting_flight_info state
    elif conversation_state == "awaiting_flight_info":
        # For simplicity, we'll just look for "yes" to confirm the cancellation
        if "yes" in user_input.lower():
            # In a real app, you'd do the cancellation and get a confirmation number
            confirmation_number = random.randint(10000, 99999) 
            response_text = f"Your flight has been successfully cancelled. Your confirmation number is {confirmation_number}. A confirmation has been sent to your email."
            return response_text, "has_email" # Return to a neutral state
        else:
            response_text = "Cancellation has been aborted. Is there anything else I can help with?"
            return response_text, "has_email" # Return to a neutral state
    # *** NEW SKILL - END ***

    # Rule 5: If we have an email, check for our keywords
    elif conversation_state == "has_email":
        user_input_lower = user_input.lower()
        # Check for order keywords
        if "order" in user_input_lower:
            return "I can help with that. Could you please provide your order number?", "awaiting_order_number"
        # Check for flight cancellation keywords
        elif "flight" in user_input_lower or "cancel" in user_input_lower:
            # *** TRIGGER the new state ***
            return "I can certainly help with that. Are you sure you want to cancel your flight? Please respond with 'yes' to confirm.", "awaiting_flight_info"
        
    # Rule 6: If no specific rules apply, THEN use the AI model
    input_seq = pad_sequences(tokenizer.texts_to_sequences([user_input]), maxlen=max_input_length, padding='post')
    model_response = decode_sequence_beam(input_seq)
    
    new_state = conversation_state

    if conversation_state == "general" and ('email' in model_response or 'dm' in model_response):
        new_state = "awaiting_email"
        
    return model_response, new_state
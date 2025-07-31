# In preprocess_data.py

import pandas as pd
from datasets import load_dataset
import re
import os

# --- NEW: Improved cleaning function that PRESERVES email characters ---
def clean_text_email_friendly(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # This regex now keeps letters, numbers, whitespace, and the '@' and '.' characters
    text = re.sub(r'[^a-zA-Z0-9\s@.]', '', text) 
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def create_conversational_pairs(df):
    inputs = []
    responses = []
    for _, row in df.iterrows():
        conversation_string = row['conversation']
        if not isinstance(conversation_string, str):
            continue
        lines = conversation_string.strip().splitlines()
        if len(lines) < 2:
            continue
        for i in range(len(lines) - 1):
            input_line = lines[i]
            response_line = lines[i+1]
            if input_line.startswith('Customer:') and response_line.startswith('Support:'):
                try:
                    # --- CHANGE: Use the new cleaning function ---
                    cleaned_input = clean_text_email_friendly(input_line.split(':', 1)[1])
                    cleaned_response = clean_text_email_friendly(response_line.split(':', 1)[1])
                    if cleaned_input and cleaned_response:
                        inputs.append(cleaned_input)
                        responses.append(cleaned_response)
                except IndexError:
                    continue
    return pd.DataFrame({'input': inputs, 'response': responses})

def main():
    print("Starting data preprocessing with the FINAL (email-friendly) script...")
    print("Loading raw dataset...")
    dataset = load_dataset("TNE-AI/customer-support-on-twitter-conversation")
    df = dataset['train'].to_pandas()
    print("Creating conversational pairs with new cleaning logic...")
    paired_conversations_df = create_conversational_pairs(df)
    
    if len(paired_conversations_df) > 0:
        print(f"Successfully created {len(paired_conversations_df)} conversational pairs.")
    else:
        print("Warning: No conversational pairs were created.")
        return

    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    output_path = os.path.join(data_dir, 'customer_support_pairs_final.csv')
    paired_conversations_df.to_csv(output_path, index=False)
    print(f"\nPreprocessing complete! Email-friendly data saved to '{output_path}'")

if __name__ == "__main__":
    main()
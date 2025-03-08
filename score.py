import os
import torch
import json
from azureml.core.model import Model
from torch import nn
from torch.jit import load
import subprocess
import sys

# Try importing torchtext, if it fails, install it
try:
    import torchtext
except ImportError:
    print("torchtext not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchtext==0.14.0"])

# Constants for special tokens
BOS_IDX = 2  # Beginning of Sentence
EOS_IDX = 3  # End of Sentence
PAD_IDX = 1  # Padding

# Initialize global variables
scripted_encoder = None
scripted_decoder = None
vocab_de = None
vocab_en = None
de_tokenizer = None
en_tokenizer = None

def init():
    global scripted_encoder, scripted_decoder, vocab_de, vocab_en, de_tokenizer, en_tokenizer

    try:
        print("Loading models and configurations...")

        # Load the encoder model
        encoder_model_path = Model.get_model_path('scripted_encoder')  # Correct registered model name
        scripted_encoder = load(encoder_model_path, map_location='cpu')
        
        # Load the decoder model
        decoder_model_path = Model.get_model_path('scripted_decoder')  # Correct registered model name
        scripted_decoder = load(decoder_model_path, map_location='cpu')
        
        # Load the configuration model (vocab, tokenizer)
        config_model_path = Model.get_model_path('model_config3')  # Correct registered model name
        config_data = torch.load(config_model_path, map_location='cpu')

        # Extract vocabularies and tokenizers from the config
        vocab_de = config_data.get('vocab_de', None)
        vocab_en = config_data.get('vocab_en', None)
        de_tokenizer = config_data.get('de_tokenizer', lambda sentence: sentence.split())
        en_tokenizer = config_data.get('en_tokenizer', lambda sentence: sentence.split())

        print("Models and configurations loaded successfully!")

    except Exception as e:
        print(f"Error during initialization: {str(e)}")

def translate_sentence(sentence, max_len=50, device='cpu'):
    """
    Translate a sentence from German to English using the trained models.
    """
    if scripted_encoder is None or scripted_decoder is None:
        print("Error: Models are not loaded.")
        return None

    scripted_encoder.eval()
    scripted_decoder.eval()

    # Tokenize and convert the input sentence in German to indices
    tokens = de_tokenizer(sentence.rstrip("\n"))  # Tokenize the German sentence
    src_ids = [vocab_de[token] for token in tokens if token in vocab_de] + [EOS_IDX]  # Convert tokens to indices and append <EOS>
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

    # Pass the source sentence through the encoder
    with torch.no_grad():
        encoder_outputs, encoder_hidden = scripted_encoder(src_tensor)

    tgt_ids = [BOS_IDX]  # Decoder input starts with <BOS>
    tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
    decoder_hidden = encoder_hidden

    output_tokens = []

    for _ in range(max_len):
        with torch.no_grad():
            y, decoder_hidden = scripted_decoder(tgt_tensor, decoder_hidden)
            next_token_id = y.argmax(2).item()

        if next_token_id == EOS_IDX:
            break

        output_tokens.append(next_token_id)
        tgt_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)

    # Convert indices back to English words using vocab_en
    translated_sentence = ' '.join([vocab_en[token] for token in output_tokens])
    return translated_sentence

def run(input_data):
    """
    The function to handle incoming requests for scoring or translation.
    """
    global scripted_encoder, scripted_decoder

    # If models are not loaded yet, call init
    if scripted_encoder is None or scripted_decoder is None:
        print("Models not loaded. Initializing...")
        init()

    try:
        input_json = json.loads(input_data)
        sentence = input_json.get('sentence', '')

        if not sentence:
            return json.dumps({'error': 'No sentence provided'})

        # Translate the sentence
        translated_sentence = translate_sentence(sentence, device='cpu')
        if translated_sentence:
            return json.dumps({'translated_sentence': translated_sentence})
        else:
            return json.dumps({'error': 'Translation failed'})

    except Exception as e:
        return json.dumps({'error': str(e)})

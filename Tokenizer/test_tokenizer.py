import sentencepiece as spm
import os

# Test texts
english_text = 'THE STUDENTS WORKED VERY HARD'
spanish_text = 'LOS ESTUDIANTES TRABAJARON MUY DURO'

# Models to test
models = {
    'BPE': '../workspace/models/tokenizer_bpe/1000_bpe.model',
    'UNIGRAM': '../workspace/models/tokenizer/1000_unigram.model',
    'CHAR': '../workspace/models/tokenizer_char/100_char.model'
}

# Test each model
for model_type, model_path in models.items():
    if not os.path.exists(model_path):
        print(f"\n{model_type}: Model not found at {model_path}")
        continue
    
    print(f"\n{'='*30}")
    print(f"MODEL: {model_type}")
    print(f"{'='*30}")
    
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    # English
    print("\n--- ENGLISH ---")
    print(f"Text: {english_text}")
    print(f"Pieces: {sp.encode_as_pieces(english_text)}")
    print(f"IDs: {sp.encode_as_ids(english_text)}")
    print(f"Decoded: {sp.decode_ids(sp.encode_as_ids(english_text))}")
    
    # Spanish
    print("\n--- SPANISH ---")
    print(f"Text: {spanish_text}")
    print(f"Pieces: {sp.encode_as_pieces(spanish_text)}")
    print(f"IDs: {sp.encode_as_ids(spanish_text)}")
    print(f"Decoded: {sp.decode_ids(sp.encode_as_ids(spanish_text))}")


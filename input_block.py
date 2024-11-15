import torch

### Step 1: Input Block ###

# Using Tiny Shakespeare dataset for character-level tokenizer. Some part of the following character-level tokenizer is referenced from Andrej karpathy's GitHub (https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py) which I found is explained very well.
# Load tiny_shakespeare data file (https://github.com/tamangmilan/llama3/blob/main/tiny_shakespeare.txt)

device: str = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Assign device to cuda or cpu based on availability

# Load tiny_shakespeare data file.
with open("tiny_shakespeare.txt", "r") as f:
    data = f.read()

# Prepare vocabulary by taking all the unique characters from the tiny_shakespeare data
vocab = sorted(list(set(data)))

# Training Llama 3 model requires addtional tokens such as <|begin_of_text|>, <|end_of_text|> and <|pad_id|>, we'll add them into vocabulary
vocab.extend(["<|begin_of_text|>", "<|end_of_text|>", "<|pad_id|>"])
vocab_size = len(vocab)

# Create a mapping between characters with corresponding integer indexes in vocabulary.
# This is important to build tokenizers encode and decode functions.
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}


# Tokenizers encode function: take a string, output a list of integers
def encode(s):
    return [stoi[ch] for ch in s]


# Tokenizers decode function: take a list of integers, output a string
def decode(l):
    return "".join(itos[i] for i in l)


# Define tensor token variable to be used later during model training
token_bos = torch.tensor([stoi["<|begin_of_text|>"]], dtype=torch.int, device=device)
token_eos = torch.tensor([stoi["<|end_of_text|>"]], dtype=torch.int, device=device)
token_pad = torch.tensor([stoi["<|pad_id|>"]], dtype=torch.int, device=device)

prompts = "Hello World"
encoded_tokens = encode(prompts)
decoded_text = decode(encoded_tokens)

### Test: Input Block Code ###
# You need take out the triple quotes below to perform testing
print(f"Lenth of shakespeare in character: {len(data)}")
print(f"The vocabulary looks like this: {''.join(vocab)}\n")
print(f"Vocab size: {vocab_size}")
print(f"encoded_tokens: {encoded_tokens}")
print(f"decoded_text: {decoded_text}")

### Test Results: ###
"""
Lenth of shakespeare in character: 1115394
The vocabulary looks like this: 
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz<|begin_of_text|><|end_of_text|><|pad_id|>

Vocab size: 68
encoded_tokens: [20, 43, 50, 50, 53, 1, 35, 53, 56, 50, 42]
decoded_text: Hello World
"""

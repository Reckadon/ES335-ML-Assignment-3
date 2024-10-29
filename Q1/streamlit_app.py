import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import re
import streamlit as st
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loadData(filepath='warpeace.txt'):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    text = text.lower()
    text = re.sub(r'[^a-z\s\.]', '', text)
    words = text.split()
    words = [word for word in words if len(word) > 1]
    words = pd.Series(words)

    unique_words = sorted(set(words))
    stoi = {s: i + 1 for i, s in enumerate(unique_words)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    vocab_size = len(stoi)

    return stoi, itos, vocab_size, unique_words, words

def load_model(embedding_size, context_length, activation_function):
    model_name = f'model-Emb{embedding_size}-Con{context_length}-{activation_function}.pth'
    model = torch.load(model_name)
    model.eval()
    return model

def preprocess_input(sentence, stoi, block_size):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-z\s\.]', '', sentence)
    words = sentence.split()
    words = [word for word in words if word]

    context = [0] * block_size
    for i in range(block_size):
        if i < len(words):
            word = words[i]
            context[i] = stoi.get(word, 0)
        else:
            context[i] = 0

    return context

def generate_sentence(model, context, itos, stoi, block_size, max_len=10):
    context = preprocess_input(context, stoi, block_size)
    words_generated = []
    
    for _ in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        word = itos[ix]
        if word == '.':
            break
        words_generated.append(word)
        context = context[1:] + [ix]

    return ' '.join(words_generated)

# Load the data
stoi, itos, vocab_size, unique_words, words = loadData()

# Streamlit app structure
st.title("Next Word Prediction")

embedding_size = st.selectbox("Select Embedding Size:", [64, 128])
context_length = st.selectbox("Select Context Length:", [5, 10])
hidden = st.selectbox("Select Hidden Size:", [1024])
activation_function = st.selectbox("Select Activation Function:", ["Tanh", "ReLU"])
num_words = st.number_input("Number of Words to Predict:", min_value=1, max_value=100)

input_text = st.text_input("Enter a sequence of words:")

if st.button("Generate Text"):
    class NextWord(nn.Module):
        def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
            self.lin2 = nn.Linear(hidden_size, vocab_size)

        def forward(self, x):
            x = self.emb(x)
            x = x.view(x.shape[0], -1)
            x = F.tanh(self.lin1(x)) if activation_function == "Tanh" else F.relu(self.lin1(x))
            x = self.lin2(x)
            return x

    model = NextWord(context_length, vocab_size, embedding_size, hidden).to(device)
    model = load_model(embedding_size, context_length, activation_function)
    st.success("Model Loaded!")

    if model is None:
        st.error("Please load a model first.")
    elif len(input_text.split()) != context_length:
        st.error(f"Please enter a sequence of {context_length} words.")
    else:
        ans = generate_sentence(model, input_text, itos, stoi, context_length, num_words)
        st.write(ans)

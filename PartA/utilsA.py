import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd

# (keep SOS_token, EOS_token, device as before)

def load_pairs(data_path, is_csv=False):
    """Load (latin, native) pairs from Dakshina TSV or preprocessed CSV"""
    pairs = []
    if is_csv:
        df = pd.read_csv(data_path)
        # Expecting columns: 'latin', 'native'
        for _, row in df.iterrows():
            pairs.append((row["latin"], row["native"]))
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                native, latin, _ = line.strip().split("\t")
                pairs.append((latin, native))
    return pairs

# ------------------------
# Global tokens & device
# ------------------------
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------
# Vocabulary
# ------------------------
class Vocab:
    def __init__(self, name):
        self.name = name
        self.char2index = {"SOS": SOS_token, "EOS": EOS_token}
        self.index2char = {SOS_token: "SOS", EOS_token: "EOS"}
        self.char2count = {}
        self.n_chars = 2

    def add_word(self, word):
        for ch in word:
            self.add_char(ch)

    def add_char(self, ch):
        if ch not in self.char2index:
            self.char2index[ch] = self.n_chars
            self.index2char[self.n_chars] = ch
            self.char2count[ch] = 1
            self.n_chars += 1
        else:
            self.char2count[ch] += 1


def prepare_vocab(pairs):
    """Build source and target vocabularies from training pairs"""
    src_vocab = Vocab("Latin")
    tgt_vocab = Vocab("Native")
    for src, tgt in pairs:
        src_vocab.add_word(src)
        tgt_vocab.add_word(tgt)
    return src_vocab, tgt_vocab


# ------------------------
# Data loading
# ------------------------
def load_pairs(data_path):
    """Load (latin, native) pairs from Dakshina TSV file"""
    pairs = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            native, latin, _ = line.strip().split("\t")
            pairs.append((latin, native))
    return pairs


# ------------------------
# Tensor conversion
# ------------------------
def word2tensor(vocab, word):
    """Convert word into tensor of indices"""
    indexes = [vocab.char2index[ch] for ch in word] + [EOS_token]
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


# ------------------------
# Encoder Model
# ------------------------
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, cell_type="LSTM"):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)

        if cell_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size)
        elif cell_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# ------------------------
# Decoder Model
# ------------------------
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, cell_type="LSTM"):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)

        if cell_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size)
        elif cell_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

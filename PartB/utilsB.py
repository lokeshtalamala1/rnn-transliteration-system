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
# Vocabulary & Data Utils
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
    src_vocab = Vocab("Latin")
    tgt_vocab = Vocab("Native")
    for src, tgt in pairs:
        src_vocab.add_word(src)
        tgt_vocab.add_word(tgt)
    return src_vocab, tgt_vocab


def load_pairs(data_path):
    pairs = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            native, latin, _ = line.strip().split("\t")
            pairs.append((latin, native))
    return pairs


def word2tensor(vocab, word):
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
        self.rnn = nn.LSTM(hidden_size, hidden_size) if cell_type == "LSTM" else nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# ------------------------
# Bahdanau Attention
# ------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        hidden = hidden.squeeze(0).repeat(seq_len, 1)  # (seq_len, hidden_size)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), 1)))
        energy = energy @ self.v
        return F.softmax(energy, dim=0).unsqueeze(0)  # (1, seq_len)


# ------------------------
# Decoder with Attention
# ------------------------
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=30, cell_type="LSTM"):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        if cell_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size * 2, hidden_size)
        else:
            self.rnn = nn.GRU(hidden_size * 2, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = self.attention(hidden[0] if isinstance(hidden, tuple) else hidden,
                                      encoder_outputs.squeeze(1))
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (1,1,hidden)

        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.rnn(rnn_input, hidden)

        output = self.out(output[0])
        return F.log_softmax(output, dim=1), hidden, attn_weights

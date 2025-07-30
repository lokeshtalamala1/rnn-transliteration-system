import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from utils import (
    load_pairs, prepare_vocab, word2tensor,
    EncoderRNN, AttnDecoderRNN,
    SOS_token, EOS_token, device
)

# ------------------------
# Training step
# ------------------------
def train_step(input_tensor, target_tensor, encoder, decoder,
               encoder_optimizer, decoder_optimizer, criterion,
               teacher_forcing_ratio=0.5, max_length=30):

    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, 1, encoder.hidden_size, device=device)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    loss = 0

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length


# ------------------------
# Training loop
# ------------------------
def train_iters(pairs, encoder, decoder, src_vocab, tgt_vocab,
                n_iters=1000, learning_rate=0.01, teacher_forcing_ratio=0.5,
                print_every=100):

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for it in range(1, n_iters + 1):
        src, tgt = random.choice(pairs)
        input_tensor = word2tensor(src_vocab, src)
        target_tensor = word2tensor(tgt_vocab, tgt)

        loss = train_step(input_tensor, target_tensor, encoder, decoder,
                          encoder_optimizer, decoder_optimizer, criterion,
                          teacher_forcing_ratio=teacher_forcing_ratio)

        if it % print_every == 0:
            print(f"Iter {it}, Loss {loss:.4f}")


# ------------------------
# Evaluation
# ------------------------
def evaluate(encoder, decoder, word, src_vocab, tgt_vocab, max_length=30):
    with torch.no_grad():
        input_tensor = word2tensor(src_vocab, word)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, 1, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_chars = []
        for di in range(max_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                break
            decoded_chars.append(tgt_vocab.index2char[topi.item()])
            decoder_input = topi.squeeze().detach()

        return ''.join(decoded_chars)


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Attention-based Seq2Seq transliteration model (Part B)")
    parser.add_argument("--data_path", type=str,
                        default="../dakshina_dataset_v1.0/hi/lexicons/hi.transliteration.train.tsv",
                        help="Path to training dataset (TSV file)")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of RNN")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--cell_type", type=str, default="LSTM", choices=["LSTM", "GRU"],
                        help="Type of RNN cell")
    parser.add_argument("--teacher_forcing", type=float, default=0.5, help="Teacher forcing ratio")
    args = parser.parse_args()

    # Load dataset
    pairs = load_pairs(args.data_path)
    src_vocab, tgt_vocab = prepare_vocab(pairs)

    # Init models
    encoder = EncoderRNN(src_vocab.n_chars, args.hidden_size, args.cell_type).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_chars, dropout_p=0.1,
                             cell_type=args.cell_type).to(device)

    print(f"Training Attention-{args.cell_type} model for {args.epochs} iterations...")
    train_iters(pairs, encoder, decoder, src_vocab, tgt_vocab,
                n_iters=args.epochs, learning_rate=args.learning_rate,
                teacher_forcing_ratio=args.teacher_forcing)

    print("\nSample Predictions:")
    for word, tgt in random.sample(pairs, 5):
        pred = evaluate(encoder, decoder, word, src_vocab, tgt_vocab)
        print(f"{word} -> {pred} (target: {tgt})")

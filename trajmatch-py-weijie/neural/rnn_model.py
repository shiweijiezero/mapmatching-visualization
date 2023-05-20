from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cpu")
torch.set_default_dtype(torch.float)

MAX_LENGTH = 5000
teacher_forcing_ratio = 0.5
import random

SOS_token = 0.0, 0.0
EOS_token = 1.0, 1.0

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size)
        )
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device).double()


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Sequential(
            nn.Linear(output_size, hidden_size * 2),
            nn.Linear(2 * hidden_size, hidden_size)
        )
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device).double()


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.embedding = nn.Sequential(
            nn.Linear(output_size, hidden_size * 2),
            nn.Linear(2 * hidden_size, hidden_size)
        )
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device).double()


def tensorFromSentence(pos_lines):
    indexes = [(x[0], x[1]) for x in pos_lines]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.double, device=device).double()


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(pair[0])
    target_tensor = tensorFromSentence(pair[1])
    return (input_tensor, target_tensor)


def infer(input_line, encoder, decoder, max_length=5000):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_line)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device).double()  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length).double()

        for di in range(max_length):
            decoder_output, \
            decoder_hidden, \
            decoder_attention = decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs
            )
            # decoder_attentions[di] = decoder_attention.data
            # topv, topi = decoder_output.data.topk(1)
            if nn.MSELoss()(decoder_input, torch.tensor([EOS_token])) < 0.5:
                break
            else:
                decoded_words.append(decoder_output[0].tolist())

            decoder_input = decoder_output

        return decoded_words

import torch

from torch.autograd import Variable
import numpy as np
from sen import SEN
import torch.nn.functional as F


class T_VAE(torch.nn.Module):

    def __init__(self, train_data, vocab_size, embedding_size, hidden_size, latent_size, num_layers, embedding_dropout, device):
        super(T_VAE, self).__init__()


        # Variables
        self.dictionary = train_data
        self.task_voc_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.num_layers = num_layers
        self.lstm_factor = num_layers

        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.device = device


        # X: bsz * seq_len * vocab_size
        # Embedding
        self.embedding = torch.nn.Embedding(num_embeddings=self.task_voc_size, embedding_dim=self.embedding_size)

        #    X: bsz * seq_len * vocab_size
        #    X: bsz * seq_len * embed_size
        self.encoder_lstm = torch.nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        self.hidden2mean = torch.nn.Linear(in_features=self.hidden_size*self.lstm_factor, out_features=self.latent_size)
        self.hidden2logv = torch.nn.Linear(in_features=self.hidden_size*self.lstm_factor, out_features=self.latent_size)


        # Decoder Part
        self.latent2hidden = torch.nn.Linear(in_features=self.latent_size, out_features=self.hidden_size*self.lstm_factor)
        self.decoder_lstm = torch.nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        self.output = torch.nn.Linear(in_features=self.hidden_size * self.lstm_factor, out_features=self.task_voc_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)


    def init_hidden(self, batch_size):
        hidden_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        state_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return (hidden_cell, state_cell)

    def get_embedding(self, x):
        x_embed = self.embedding(x)

        # Total length for pad_packed_sequence method = maximum sequence length
        maximum_sequence_length = x_embed.size(1)
        return x_embed, maximum_sequence_length

    def encoder(self, packed_x_embed, total_padding_length, hidden_encoder):

        # pad the packed input.

        packed_output_encoder, hidden_encoder = self.encoder_lstm(packed_x_embed, hidden_encoder)   # hidden_encoder: 32 *128

        output_encoder, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output_encoder, batch_first=True,
                                                                   total_length=total_padding_length)
        # Extimate the mean and the variance of q(z|x)


        mean = self.hidden2mean(hidden_encoder[0])
        log_var = self.hidden2logv(hidden_encoder[0])
        std = torch.exp(0.5 * log_var)  # e^(0.5 log_var) = var^0.5

        # Generate a unit gaussian noise.
        batch_size = output_encoder.size(0)
        noise = torch.randn(batch_size, self.latent_size).to(self.device)
        z = noise * std + mean   # 32 * 50
        return z, mean, log_var, hidden_encoder

    def decoder(self, z, packed_x_embed, total_padding_length=None):

        hidden_decoder = self.latent2hidden(z)

        hidden_decoder = (hidden_decoder, hidden_decoder)

        # pad the packed input.
        packed_output_decoder, hidden_decoder = self.decoder_lstm(packed_x_embed, hidden_decoder)

        output_decoder, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output_decoder, batch_first=True,
                                                                   total_length=total_padding_length)

        x_hat = self.output(output_decoder)

        x_hat = self.log_softmax(x_hat)

        return x_hat

    def forward(self, x, sentences_length, hidden_encoder):


        # Get Embeddings
        x_embed, maximum_padding_length = self.get_embedding(x)    # x: 32 * 60  x_embed: 32 * 60 * 300  maximum_padding_length:60

        # Packing the input
        packed_x_embed = torch.nn.utils.rnn.pack_padded_sequence(input=x_embed, lengths=sentences_length.data.tolist(), batch_first=True, enforce_sorted=False)

        # Encoder
        z, mean, log_var, hidden_encoder = self.encoder(packed_x_embed, maximum_padding_length, hidden_encoder)

        # Decoder
        x_embed = self.embedding_dropout(x_embed)

        packed_x_embed = torch.nn.utils.rnn.pack_padded_sequence(input=x_embed, lengths=sentences_length.data.tolist(), batch_first=True, enforce_sorted=False)

        x_hat = self.decoder(z, packed_x_embed, maximum_padding_length)

        return x_hat, mean, log_var, z, hidden_encoder


    def inference(self, n_samples, sos, z):
        # generate random z
        batch_size = 1
        seq_len = 1
        idx_sample = []
        input = torch.Tensor(1, 1).fill_(self.dictionary.get_w2i()[sos]).long().to(self.device)


        hidden = self.latent2hidden(z)
        hidden = (hidden, hidden)

        for i in range(n_samples):
            input = self.embedding(input)
            output, hidden = self.decoder_lstm(input, hidden)
            output = self.output(output)
            output = self.log_softmax(output)
            output = output.exp()
            _, s = torch.topk(output, 1)
            idx_sample.append(s.item())
            input = s.squeeze(0)

        w_sample = [self.dictionary.get_i2w()[str(idx)] for idx in idx_sample]
        w_sample = " ".join(w_sample)

        return w_sample

class A_VAE(torch.nn.Module):

    def __init__(self, E_in, middle_size, hidden_size, latent_size, D_out, device):
        super().__init__()


        self.E_in = E_in
        self.D_out = D_out
        self.middle_size = middle_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.device = device

        self.E_linear1 = torch.nn.Linear(self.E_in, self.middle_size)
        self.E_linear2 = torch.nn.Linear(self.middle_size, self.hidden_size)

        self.hidden2mean = torch.nn.Linear(self.hidden_size, self.latent_size)
        self.hidden2logv = torch.nn.Linear(self.hidden_size, self.latent_size)

        self.latent2hidden = torch.nn.Linear(self.latent_size, self.hidden_size)
        self.D_linear1 = torch.nn.Linear(self.hidden_size, self.middle_size)
        self.D_linear2 = torch.nn.Linear(self.middle_size, self.D_out)


    def forward(self, input):

        batch_size = input.size(0)
        # ENCODER
        e_middle = torch.nn.functional.relu(self.E_linear1(input))
        hidden = torch.nn.functional.relu(self.E_linear2(e_middle))

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)
        z = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)
        d_middle = torch.nn.functional.relu(self.D_linear1(hidden))
        ouput = torch.nn.functional.relu(self.D_linear2(d_middle))
        return ouput, mean, logv, z



class My_Model(torch.nn.Module):
    def __init__(self, a_vae, t_vae):
        super().__init__()
        self.a_vae = a_vae
        self.t_vae = t_vae
        # self.bn1 = torch.nn.BatchNorm1d(58)
        # self.linear1 = torch.nn.Linear(58, 24)
        # self.bn2 = torch.nn.BatchNorm1d(24)
        # self.linear2 = torch.nn.Linear(24, 1)

        self.hidden = torch.nn.Linear(58, 16)
        self.output = torch.nn.Linear(16, 2)
    def forward(self, x):
        # x = self.bn1(x)
        # x = torch.nn.functional.relu(self.linear1(x))
        # x = self.bn2(x)
        # x = torch.nn.functional.relu(self.linear2(x))
        # x = self.Sig(x)
        x = torch.nn.functional.relu(self.hidden(x))
        x = torch.nn.functional.relu(self.output(x))
        return x





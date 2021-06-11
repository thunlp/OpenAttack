"""
:type: tuple
:Size: 55.041MB
:Package Requirements:
    * **pytorch**

Pretrained GAN model on SNLI dataset used in :py:class:`.GANAttacker`. See :py:class:`.GANAttacker` for detail.
"""

import os
from OpenAttack.utils import make_zip_downloader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
from torch.autograd import Variable

NAME = "AttackAssist.GAN"
URL = "/TAADToolbox/GNAE.zip"
DOWNLOAD = make_zip_downloader(URL)

try:
    
    def to_gpu(_, x):
        return x
    class MLP_D(nn.Module):
        def __init__(self, ninput, noutput, layers,
                     activation=nn.LeakyReLU(0.2), gpu=False):
            super(MLP_D, self).__init__()
            self.ninput = ninput
            self.noutput = noutput

            layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
            self.layers = []

            for i in range(len(layer_sizes) - 1):
                layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                self.layers.append(layer)
                self.add_module("layer" + str(i + 1), layer)

                # No batch normalization after first layer
                if i != 0:
                    bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
                    self.layers.append(bn)
                    self.add_module("bn" + str(i + 1), bn)

                self.layers.append(activation)
                self.add_module("activation" + str(i + 1), activation)

            layer = nn.Linear(layer_sizes[-1], noutput)
            self.layers.append(layer)
            self.add_module("layer" + str(len(self.layers)), layer)

            self.init_weights()

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
            x = torch.mean(x)
            return x

        def init_weights(self):
            init_std = 0.02
            for layer in self.layers:
                try:
                    layer.weight.data.normal_(0, init_std)
                    layer.bias.data.fill_(0)
                except:
                    pass

    class MLP_G(nn.Module):
        def __init__(self, ninput, noutput, layers,
                     activation=nn.ReLU(), gpu=False):
            super(MLP_G, self).__init__()
            self.ninput = ninput
            self.noutput = noutput
            self.gpu = gpu

            layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
            self.layers = []

            for i in range(len(layer_sizes) - 1):
                layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                self.layers.append(layer)
                self.add_module("layer" + str(i + 1), layer)

                bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn" + str(i + 1), bn)

                self.layers.append(activation)
                self.add_module("activation" + str(i + 1), activation)

            layer = nn.Linear(layer_sizes[-1], noutput)
            self.layers.append(layer)
            self.add_module("layer" + str(len(self.layers)), layer)

            self.init_weights()

        def forward(self, x):
            if x.__class__.__name__ == "ndarray":
                x = Variable(torch.FloatTensor(x)).cuda()
                # x = x.cpu()
            if x.__class__.__name__ == "FloatTensor":
                x = Variable(x).cuda()
            for i, layer in enumerate(self.layers):
                x = layer(x)
            return x

        def init_weights(self):
            init_std = 0.02
            for layer in self.layers:
                try:
                    layer.weight.data.normal_(0, init_std)
                    layer.bias.data.fill_(0)
                except:
                    pass

    class MLP_I(nn.Module):
        # separate Inverter to map continuous code back to z  inverter，从continuous->z?
        def __init__(self, ninput, noutput, layers,
                     activation=nn.ReLU(), gpu=False):
            super(MLP_I, self).__init__()
            self.ninput = ninput
            self.noutput = noutput

            layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
            self.layers = []

            for i in range(len(layer_sizes) - 1):
                layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                self.layers.append(layer)
                self.add_module("layer" + str(i + 1), layer)

                bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn" + str(i + 1), bn)

                self.layers.append(activation)
                self.add_module("activation" + str(i + 1), activation)

            layer = nn.Linear(layer_sizes[-1], noutput)
            self.layers.append(layer)
            self.add_module("layer" + str(len(self.layers)), layer)

            self.init_weights()

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
            return x

        def init_weights(self):
            init_std = 0.02
            for layer in self.layers:
                try:
                    layer.weight.data.normal_(0, init_std)
                    layer.bias.data.fill_(0)
                except:
                    pass

    class MLP_I_AE(nn.Module):
        # separate Inverter to map continuous code back to z (mean & std)
        def __init__(self, ninput, noutput, layers,
                     activation=nn.ReLU(), gpu=False):
            super(MLP_I_AE, self).__init__()
            self.ninput = ninput
            self.noutput = noutput
            self.gpu = gpu
            noutput_mu = noutput
            noutput_var = noutput

            layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
            self.layers = []

            for i in range(len(layer_sizes) - 1):
                layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                self.layers.append(layer)
                self.add_module("layer" + str(i + 1), layer)

                bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn" + str(i + 1), bn)

                self.layers.append(activation)
                self.add_module("activation" + str(i + 1), activation)

            layer = nn.Linear(layer_sizes[-1], noutput)
            self.layers.append(layer)
            self.add_module("layer" + str(len(self.layers)), layer)

            self.linear_mu = nn.Linear(noutput, noutput_mu)
            self.linear_var = nn.Linear(noutput, noutput_var)

            self.init_weights()

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
            mu = self.linear_mu(x)
            logvar = self.linear_var(x)
            std = 0.5 * logvar
            std = std.exp_()  # std
            epsilon = Variable(
                std.data.new(std.size()).normal_())  # normal noise with the same type and size as std.data
            if self.gpu:
                epsilon = epsilon.cuda()

            sample = mu + (epsilon * std)

            return sample

        def init_weights(self):
            init_std = 0.02
            for layer in self.layers:
                try:
                    layer.weight.data.normal_(0, init_std)
                    layer.bias.data.fill_(0)
                except:
                    pass

            self.linear_mu.weight.data.normal_(0, init_std)
            self.linear_mu.bias.data.fill_(0)
            self.linear_var.weight.data.normal_(0, init_std)
            self.linear_var.bias.data.fill_(0)

    class Seq2SeqCAE(nn.Module):
        # CNN encoder, LSTM decoder
        def __init__(self, emsize, nhidden, ntokens, nlayers, conv_windows="5-5-3", conv_strides="2-2-2",
                     conv_layer="500-700-1000", activation=nn.LeakyReLU(0.2, inplace=True),
                     noise_radius=0.2, hidden_init=False, dropout=0, gpu=True):
            super(Seq2SeqCAE, self).__init__()
            self.nhidden = nhidden  # size of hidden vector in LSTM
            self.emsize = emsize
            self.ntokens = ntokens
            self.nlayers = nlayers
            self.noise_radius = noise_radius
            self.hidden_init = hidden_init
            self.dropout = dropout
            self.gpu = gpu
            self.arch_conv_filters = conv_layer
            self.arch_conv_strides = conv_strides
            self.arch_conv_windows = conv_windows
            self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

            # Vocabulary embedding
            self.embedding = nn.Embedding(ntokens, emsize)
            self.embedding_decoder = nn.Embedding(ntokens, emsize)

            conv_layer_sizes = [emsize] + [int(x) for x in conv_layer.split('-')]
            conv_strides_sizes = [int(x) for x in conv_strides.split('-')]
            conv_windows_sizes = [int(x) for x in conv_windows.split('-')]
            self.encoder = nn.Sequential()

            for i in range(len(conv_layer_sizes) - 1):
                layer = nn.Conv1d(conv_layer_sizes[i], conv_layer_sizes[i + 1], \
                                  conv_windows_sizes[i], stride=conv_strides_sizes[i])
                self.encoder.add_module("layer-" + str(i + 1), layer)

                bn = nn.BatchNorm1d(conv_layer_sizes[i + 1])
                self.encoder.add_module("bn-" + str(i + 1), bn)

                self.encoder.add_module("activation-" + str(i + 1), activation)

            self.linear = nn.Linear(1000, emsize)

            decoder_input_size = emsize + nhidden
            self.decoder = nn.LSTM(input_size=decoder_input_size,
                                   hidden_size=nhidden,
                                   num_layers=1,
                                   dropout=dropout,
                                   batch_first=True)
            self.linear_dec = nn.Linear(nhidden, ntokens)

            # 9-> 7-> 3 -> 1

        def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
            # batch x hidden
            all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

            if self.hidden_init:
                # initialize decoder hidden state to encoder output
                state = (hidden.unsqueeze(0), self.init_state(batch_size))
            else:
                state = self.init_hidden(batch_size)

            embeddings = self.embedding_decoder(indices)  # training stage
            augmented_embeddings = torch.cat([embeddings, all_hidden], 2)

            output, state = self.decoder(augmented_embeddings, state)

            decoded = self.linear_dec(output.contiguous().view(-1, self.nhidden))
            decoded = decoded.view(batch_size, maxlen, self.ntokens)

            return decoded

        def generate(self, hidden, maxlen, sample=True, temp=1.0):
            """Generate through decoder; no backprop"""
            if hidden.ndimension() == 1:
                hidden = hidden.unsqueeze(0)
            batch_size = hidden.size(0)

            if self.hidden_init:
                # initialize decoder hidden state to encoder output
                state = (hidden.unsqueeze(0), self.init_state(batch_size))
            else:
                state = self.init_hidden(batch_size)

            if not self.gpu:
                self.start_symbols = self.start_symbols.cpu()
            # <sos>
            # self.start_symbols.data.resize_(batch_size, 1)
            with torch.no_grad():
                self.start_symbols.resize_(batch_size, 1)
            # self.start_symbols.data.fill_(1)
            with torch.no_grad():
                self.start_symbols.fill_(1)

            embedding = self.embedding_decoder(self.start_symbols)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

            # unroll
            all_indices = []
            for i in range(maxlen):
                output, state = self.decoder(inputs, state)
                overvocab = self.linear_dec(output.squeeze(1))

                if not sample:
                    vals, indices = torch.max(overvocab, 1)
                else:
                    # sampling
                    probs = F.softmax(overvocab / temp, dim=1)
                    indices = torch.multinomial(probs, 1)

                if indices.ndimension() == 1:
                    indices = indices.unsqueeze(1)
                all_indices.append(indices)

                embedding = self.embedding_decoder(indices)
                inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

            max_indices = torch.cat(all_indices, 1)

            return max_indices

        def init_weights(self):
            initrange = 0.1

            # Initialize Vocabulary Matrix Weight
            self.embedding.weight.data.uniform_(-initrange, initrange)
            self.embedding.weight.data[0].zero()
            self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

            # Initialize Encoder and Decoder Weights
            for p in self.encoder.parameters():
                p.data.uniform_(-initrange, initrange)
            for p in self.decoder.parameters():
                p.data.uniform_(-initrange, initrange)

            # Initialize Linear Weight
            self.linear.weight.data.uniform_(-initrange, initrange)
            self.linear.bias.data.fill_(0)

        def encode(self, indices, lengths, noise):
            embeddings = self.embedding(indices)
            embeddings = embeddings.transpose(1, 2)
            c_pre_lin = self.encoder(embeddings)
            c_pre_lin = c_pre_lin.squeeze(2)
            hidden = self.linear(c_pre_lin)
            # normalize to unit ball (l2 norm of 1) - p=2, dim=1
            norms = torch.norm(hidden, 2, 1)
            if norms.ndimension() == 1:
                norms = norms.unsqueeze(1)
            hidden = torch.div(hidden, norms.expand_as(hidden))

            if noise and self.noise_radius > 0:
                gauss_noise = torch.normal(mean=torch.zeros(hidden.size()),
                                           std=self.noise_radius, generator=torch.Generator(), out=None)  ###
                if self.gpu:
                    gauss_noise = gauss_noise.cuda()

                hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

            return hidden

        def init_hidden(self, bsz):
            zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
            zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
            return to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2)  # (hidden, cell)

        def init_state(self, bsz):
            zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
            return to_gpu(self.gpu, zeros)

        def store_grad_norm(self, grad):
            norm = torch.norm(grad, 2, 1)
            self.grad_norm = norm.detach().data.mean()
            return grad

        def forward(self, indices, lengths, noise, encode_only=False, generator=None, inverter=None):
            if not generator:  # only enc -> dec
                batch_size, maxlen = indices.size()
                self.embedding.weight.data[0].fill_(0)
                self.embedding_decoder.weight.data[0].fill_(0)
                hidden = self.encode(indices, lengths, noise)
                if encode_only:
                    return hidden

                if hidden.requires_grad:
                    hidden.register_hook(self.store_grad_norm)

                decoded = self.decode(hidden, batch_size, maxlen,
                                      indices=indices, lengths=lengths)
            else:  # enc -> inv -> gen -> dec
                batch_size, maxlen = indices.size()
                self.embedding.weight.data[0].fill_(0)
                self.embedding_decoder.weight.data[0].fill_(0)
                hidden = self.encode(indices, lengths, noise)
                if encode_only:
                    return hidden

                if hidden.requires_grad:
                    hidden.register_hook(self.store_grad_norm)

                z_hat = inverter(hidden)
                c_hat = generator(z_hat)

                decoded = self.decode(c_hat, batch_size, maxlen,
                                      indices=indices, lengths=lengths)

            return decoded

    class Seq2Seq(nn.Module):
        def __init__(self, emsize, nhidden, ntokens, nlayers, noise_radius=0.2,
                     hidden_init=False, dropout=0, gpu=True):
            super(Seq2Seq, self).__init__()
            self.nhidden = nhidden
            self.emsize = emsize
            self.ntokens = ntokens
            self.nlayers = nlayers
            self.noise_radius = noise_radius
            self.hidden_init = hidden_init
            self.dropout = dropout
            self.gpu = gpu

            self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

            # Vocabulary embedding
            self.embedding = nn.Embedding(ntokens, emsize)
            self.embedding_decoder = nn.Embedding(ntokens, emsize)

            # RNN Encoder and Decoder
            self.encoder = nn.LSTM(input_size=emsize,
                                   hidden_size=nhidden,
                                   num_layers=nlayers,
                                   dropout=dropout,
                                   batch_first=True)

            decoder_input_size = emsize + nhidden
            self.decoder = nn.LSTM(input_size=decoder_input_size,
                                   hidden_size=nhidden,
                                   num_layers=1,
                                   dropout=dropout,
                                   batch_first=True)

            # Initialize Linear Transformation
            self.linear = nn.Linear(nhidden, ntokens)

            self.init_weights()

        def init_weights(self):
            initrange = 0.1

            # Initialize Vocabulary Matrix Weight
            self.embedding.weight.data.uniform_(-initrange, initrange)
            self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

            # Initialize Encoder and Decoder Weights
            for p in self.encoder.parameters():
                p.data.uniform_(-initrange, initrange)
            for p in self.decoder.parameters():
                p.data.uniform_(-initrange, initrange)

            # Initialize Linear Weight
            self.linear.weight.data.uniform_(-initrange, initrange)
            self.linear.bias.data.fill_(0)

        def init_hidden(self, bsz):
            zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
            zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
            return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))  # (hidden, cell)

        def init_state(self, bsz):
            zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
            return to_gpu(self.gpu, zeros)

        def store_grad_norm(self, grad):
            norm = torch.norm(grad, 2, 1)
            self.grad_norm = norm.detach().data.mean()
            return grad

        def forward(self, indices, lengths, noise, encode_only=False, generator=None, inverter=None):
            if not generator:
                batch_size, maxlen = indices.size()

                hidden = self.encode(indices, lengths, noise)

                if encode_only:
                    return hidden

                if hidden.requires_grad:
                    hidden.register_hook(self.store_grad_norm)

                decoded = self.decode(hidden, batch_size, maxlen,
                                      indices=indices, lengths=lengths)
            else:
                batch_size, maxlen = indices.size()
                self.embedding.weight.data[0].fill_(0)
                self.embedding_decoder.weight.data[0].fill_(0)
                hidden = self.encode(indices, lengths, noise)
                if encode_only:
                    return hidden

                if hidden.requires_grad:
                    hidden.register_hook(self.store_grad_norm)

                z_hat = inverter(hidden)
                c_hat = generator(z_hat)

                decoded = self.decode(c_hat, batch_size, maxlen,
                                      indices=indices, lengths=lengths)

            return decoded

        def encode(self, indices, lengths, noise):
            embeddings = self.embedding(indices)
            packed_embeddings = pack_padded_sequence(input=embeddings,
                                                     lengths=lengths,
                                                     batch_first=True)

            # Encode
            packed_output, state = self.encoder(packed_embeddings)

            hidden, cell = state
            # batch_size x nhidden
            hidden = hidden[-1]  # get hidden state of last layer of encoder

            # normalize to unit ball (l2 norm of 1) - p=2, dim=1
            norms = torch.norm(hidden, 2, 1)
            if norms.ndimension() == 1:
                norms = norms.unsqueeze(1)
            hidden = torch.div(hidden, norms.expand_as(hidden))

            if noise and self.noise_radius > 0:
                gauss_noise = torch.normal(means=torch.zeros(hidden.size()),
                                           std=self.noise_radius)
                hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

            return hidden

        def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
            # batch x hidden
            all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

            if self.hidden_init:
                # initialize decoder hidden state to encoder output
                state = (hidden.unsqueeze(0), self.init_state(batch_size))
            else:
                state = self.init_hidden(batch_size)

            embeddings = self.embedding_decoder(indices)
            augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
            packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                     lengths=lengths,
                                                     batch_first=True)

            packed_output, state = self.decoder(packed_embeddings, state)
            output, lengths = pad_packed_sequence(packed_output, batch_first=True)

            # reshape to batch_size*maxlen x nhidden before linear over vocab
            decoded = self.linear(output.contiguous().view(-1, self.nhidden))
            decoded = decoded.view(batch_size, maxlen, self.ntokens)

            return decoded

        def generate(self, hidden, maxlen, sample=True, temp=1.0):
            """Generate through decoder; no backprop"""

            batch_size = hidden.size(0)

            if self.hidden_init:
                # initialize decoder hidden state to encoder output
                state = (hidden.unsqueeze(0), self.init_state(batch_size))
            else:
                state = self.init_hidden(batch_size)

            # <sos>
            self.start_symbols.data.resize_(batch_size, 1)
            self.start_symbols.data.fill_(1)

            embedding = self.embedding_decoder(self.start_symbols)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

            # unroll
            all_indices = []
            for i in range(maxlen):
                output, state = self.decoder(inputs, state)
                overvocab = self.linear(output.squeeze(1))

                if not sample:
                    vals, indices = torch.max(overvocab, 1)
                else:
                    # sampling
                    probs = F.softmax(overvocab / temp)
                    indices = torch.multinomial(probs, 1)

                if indices.ndimension() == 1:
                    indices = indices.unsqueeze(1)
                all_indices.append(indices)

                embedding = self.embedding_decoder(indices)
                inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

            max_indices = torch.cat(all_indices, 1)
            return max_indices


    def LOAD(path):
        word2idx = json.load(open(os.path.join(path, 'vocab.json'), 'r'))
        ntokens = len(word2idx)
        autoencoder = Seq2SeqCAE(emsize=300,
                                nhidden=300,
                                ntokens=ntokens,
                                nlayers=1,
                                noise_radius=0.2,
                                hidden_init=False,
                                dropout=0.0,
                                conv_layer='500-700-1000',
                                conv_windows='3-3-3',
                                conv_strides='1-2-2',
                                gpu=False)
        inverter = MLP_I_AE(ninput=300, noutput=100, layers='300-300')
        gan_gen = MLP_G(ninput=100, noutput=300, layers='300-300')
        gan_disc = MLP_D(ninput=300, noutput=1, layers='300-300')
        autoencoder.load_state_dict(torch.load(os.path.join(path, 'a.pkl')))
        inverter.load_state_dict(torch.load(os.path.join(path, 'i.pkl')))
        gan_gen.load_state_dict(torch.load(os.path.join(path, 'g.pkl')))
        gan_disc.load_state_dict(torch.load(os.path.join(path, 'd.pkl')))

        return word2idx, autoencoder, inverter, gan_gen, gan_disc
except ModuleNotFoundError as e:
    def LOAD(path):
        raise e
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
import torchaudio
import torchaudio.transforms as tat
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pytorch_audio_utils import *

parser = argparse.ArgumentParser(description='PyTorch Language ID Classifier Trainer')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=6,
                    help='batch size')
parser.add_argument('--window-size', type=int, default=200,
                    help='size of fft window')
parser.add_argument('--validate', action='store_true',
                    help='do out-of-bag validation')
parser.add_argument('--log-interval', type=int, default=5,
                    help='reports per epoch')
parser.add_argument('--load-model', type=str, default=None,
                    help='path of model to load')
parser.add_argument('--save-model', action='store_true',
                    help='path to save the final model')
parser.add_argument('--train-full-model', action='store_true',
                    help='train full model vs. final layer')
args = parser.parse_args()

class Preemphasis(object):
    """Perform preemphasis on signal

    y = x[n] - α*x[n-1]

    Args:
        alpha (float): preemphasis coefficient

    """

    def __init__(self, alpha=0.97):
        self.alpha = alpha

    def __call__(self, sig):
        """

        Args:
            sig (Tensor): Tensor of audio of size (Samples x Channels)

        Returns:
            sig (Tensor): Preemphasized. See equation above.

        """
        if self.alpha == 0:
            return sig
        else:
            sig[1:, :] -= self.alpha * sig[:-1, :]
            return sig

class RfftPow(object):
    """This function emulates power of the discrete fourier transform.

    Note: this implementation may not be numerically stable

    Args:
        K (int): number of fft freq bands

    """

    def __init__(self, K=None):
        self.K = K

    def __call__(self, sig):
        """

        Args:
            sig (Tensor): Tensor of audio of size (Samples x Channels)

        Returns:
            S (Tensor): spectrogram

        """
        N = sig.size(1)
        if self.K is None:
            K = N
        else:
            K = self.K

        k_vec = torch.arange(0, K).unsqueeze(0)
        n_vec = torch.arange(0, N).unsqueeze(1)
        angular_pt = 2 * np.pi * k_vec * n_vec / K
        S = torch.sqrt(torch.matmul(sig, angular_pt.cos())**2 + \
                       torch.matmul(sig, angular_pt.sin())**2)
        S = S.squeeze()[:(K//2+1)]
        S = (1 / K) * S**2
        return S

class FilterBanks(object):
    """Bins a periodogram from K fft frequency bands into N bins (banks)

    fft bands (K//2+1) -> filterbanks (n_filterbanks) -> bins (bins)

    Args:
        n_filterbanks (int): number of filterbanks
        bins (list): number of bins

    """

    def __init__(self, n_filterbanks, bins):
        self.n_filterbanks = n_filterbanks
        self.bins = bins

    def __call__(self, S):
        """

        Args:
            S (Tensor): Tensor of Spectro- / Periodogram

        Returns:
            fb (Tensor): binned filterbanked spectrogram

        """
        conversion_factor = np.log(10) # torch.log10 doesn't exist
        K = S.size(0)
        fb_mat = torch.zeros((self.n_filterbanks, K))
        for m in range(1, self.n_filterbanks+1):
            f_m_minus = int(self.bins[m - 1])
            f_m = int(self.bins[m])
            f_m_plus = int(self.bins[m + 1])

            fb_mat[m - 1, f_m_minus:f_m] = (torch.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
            fb_mat[m - 1, f_m:f_m_plus] = (f_m_plus - torch.arange(f_m, f_m_plus)) / (f_m_plus - f_m)
        fb = torch.matmul(S, fb_mat.t())
        fb = 20 * torch.log(fb) / conversion_factor
        return fb

class MFCC(object):
    """Discrete Cosine Transform

    There are three types of the DCT.  This is 'Type 2' as described in the scipy docs.

    filterbank bins (bins) -> mfcc (mfcc)

    Args:
        n_filterbanks (int): number of filterbanks
        n_coeffs (int): number of mfc coefficients to keep
        mode (str): orthogonal transformation

    """

    def __init__(self, n_filterbanks, n_coeffs, mode="ortho"):
        self.n_filterbanks = n_filterbanks
        self.n_coeffs = n_coeffs
        self.mode = "ortho"

    def __call__(self, fb):
        """

        Args:
            fb (Tensor): Tensor of binned filterbanked spectrogram

        Returns:
            mfcc (Tensor): Tensor of mfcc coefficients

        """
        K = self.n_filterbanks
        k_vec = torch.arange(0, K).unsqueeze(0)
        n_vec = torch.arange(0, self.n_filterbanks).unsqueeze(1)
        angular_pt = np.pi * k_vec * ((2*n_vec+1) / (2*K))
        mfcc = 2 * torch.matmul(fb, angular_pt.cos())
        if self.mode == "ortho":
            mfcc[0] *= np.sqrt(1/(4*self.n_filterbanks))
            mfcc[1:] *= np.sqrt(1/(2*self.n_filterbanks))
        return mfcc[1:(self.n_coeffs+1)]

class Sig2Features(object):
    """Get the log power, MFCCs and 1st derivatives of the signal across n hops
    and concatenate all that together

    Args:
        n_hops (int): number of filterbanks
        transformDict (dict): dict of transformations for each hop

    """

    def __init__(self, ws, hs, transformDict):
        self.ws = ws
        self.hs = hs
        self.td = transformDict

    def __call__(self, sig):
        """

        Args:
            sig (Tensor): Tensor of signal

        Returns:
            Feats (Tensor): Tensor of log-power, 12 mfcc coefficients and 1st devs

        """
        n_hops = (sig.size(0) - ws) // hs

        P = []
        Mfcc = []

        for i in range(n_hops):
            # create frame
            st = int(i * hs)
            end = st + ws
            sig_n = sig[st:end]

            # get power/energy
            P += [self.td["RfftPow"](sig_n.transpose(0, 1))]

            # get mfccs and filter banks
            fb = self.td["FilterBanks"](P[-1])
            Mfcc += [self.td["MFCC"](fb)]

        # concat and calculate derivatives
        P = torch.stack(P, 1)
        P_sum = torch.log(P.sum(0))
        P_dev = torch.zeros(P_sum.size())
        P_dev[1:] = P_sum[1:] - P_sum[:-1]
        Mfcc = torch.stack(Mfcc, 1)
        Mfcc_dev = torch.cat((torch.zeros(n_coefficients, 1), Mfcc[:,:-1] - Mfcc[:,1:]), 1)
        Feats = torch.cat((P_sum.unsqueeze(0), P_dev.unsqueeze(0), Mfcc, Mfcc_dev), 0)
        return Feats

class Labeler(object):
    """Labels from text to int + 1

    """

    def __call__(self, labels):
        return torch.LongTensor([int(l)+1 for l in labels])

def pad_packed_collate(batch):
    """Puts data, and lengths into a packed_padded_sequence then returns
       the packed_padded_sequence and the labels. Set use_lengths to True
       to use this collate function.

       Args:
         batch: (list of tuples) [(audio, target)].
             audio is a FloatTensor
             target is a LongTensor with a length of 8
       Output:
         packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
         labels: (Tensor), labels from the file names of the wav.

    """

    if len(batch) == 1:
        sigs, labels = batch[0][0], batch[0][1]
        sigs = sigs.t()
        lengths = [sigs.size(0)]
        sigs.unsqueeze_(0)
        labels.unsqueeze_(0)
    if len(batch) > 1:
        sigs, labels, lengths = zip(*[(a.t(), b, a.size(1)) for (a,b) in sorted(batch, key=lambda x: x[0].size(1), reverse=True)])
        max_len, n_feats = sigs[0].size()
        sigs = [torch.cat((s, torch.zeros(max_len - s.size(0), n_feats)), 0) if s.size(0) != max_len else s for s in sigs]
        sigs = torch.stack(sigs, 0)
        labels = torch.stack(labels, 0)
    packed_batch = pack(Variable(sigs), lengths, batch_first=True)
    return packed_batch, labels

def unpack_lengths(batch_sizes):
    """taken directly from pad_packed_sequence()
    """
    lengths = []
    data_offset = 0
    prev_batch_size = batch_sizes[0]
    for i, batch_size in enumerate(batch_sizes):
        dec = prev_batch_size - batch_size
        if dec > 0:
            lengths.extend((i,) * dec)
        prev_batch_size = batch_size
    lengths.extend((i + 1,) * batch_size)
    lengths.reverse()
    return lengths

class EncoderRNN2(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, batch_size=1):
        super(EncoderRNN2, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)

    def forward(self, input, hidden):
        output = input
        output, hidden = self.gru(output, hidden)
        #print("encoder:", output.size(), hidden.size())
        return output, hidden

    def initHidden(self, ttype=None):
        if ttype == None:
            ttype = torch.FloatTensor
        result = Variable(ttype(self.n_layers * 1, self.batch_size, self.hidden_size).fill_(0))
        if use_cuda:
            return result.cuda()
        else:
            return result

class Attn(nn.Module):
    def __init__(self, hidden_size, batch_size=1, method="dot"):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size, bias=False)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Parameter(torch.FloatTensor(batch_size, 1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)

        # get attn energies in one batch
        attn_energies = self.score(hidden, encoder_outputs)

        # Normalize energies to weights in range 0 to 1
        return F.softmax(attn_energies)

    def score(self, hidden, encoder_output):
        #print("attn.score:", hidden.size(), encoder_output.size())
        if self.method == 'general':
            energy = self.attn(encoder_output)
            energy = energy.transpose(2, 1)
            energy = hidden.bmm(energy)
            return energy

        elif self.method == 'concat':
            hidden = hidden * Variable(encoder_output.data.new(encoder_output.size()).fill_(1)) # broadcast hidden to encoder_outputs size
            energy = self.attn(torch.cat((hidden, encoder_output), -1))
            energy = energy.transpose(2, 1)
            energy = self.v.bmm(energy)
            return energy
        else:
            #self.method == 'dot':
            encoder_output = encoder_output.transpose(2, 1)
            energy = hidden.bmm(encoder_output)
            return energy

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, attn_model="dot", n_layers=1, dropout=0.1, batch_size=1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size

        # Define layers
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(hidden_size, method=attn_model, batch_size=batch_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: This now runs in batch but was originally run one
        #       step at a time
        #       B = batch size
        #       S = output length
        #       N = # of hidden features

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(input_seq, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        #print("decoder:", rnn_output.size(), encoder_outputs.size())
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs) # [B, S, L] dot [B, L, N] -> [B, S, N]
        print(attn_weights.size(), encoder_outputs.size(), context.size())
        #print("decoder context:", context.size())

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        concat_input = torch.cat((rnn_output, context), -1) # B x S x 2*N
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

# train parameters
epochs = args.epochs

# set dataset parameters
DATADIR = "/home/david/Programming/data"
sr = 8000
ws = args.window_size
hs = ws // 2
n_fft = 512 # 256
n_filterbanks = 26
n_coefficients = 12
low_mel_freq = 0
high_freq_mel = (2595 * np.log10(1 + (sr/2) / 700))
mel_pts = np.linspace(low_mel_freq, high_freq_mel, n_filterbanks + 2)
hz_pts = np.floor(700 * (10**(mel_pts / 2595) - 1))
bins = np.floor((n_fft + 1) * hz_pts / sr)

# data transformations
td = {
    "RfftPow": RfftPow(n_fft),
    "FilterBanks": FilterBanks(n_filterbanks, bins),
    "MFCC": MFCC(n_filterbanks, n_coefficients),
}

transforms = tat.Compose([
                 tat.Scale(),
                 tat.PadTrim(58000, fill_value=1e-8),
                 Preemphasis(),
                 Sig2Features(ws, hs, td),
             ])

# set network parameters
use_cuda = torch.cuda.is_available()
batch_size = args.batch_size
input_features = 26
hidden_size = 100
output_size = 3
#output_length = (8 + 7 + 2) # with "blanks"
output_length = 8 # without blanks
n_layers = 1
attn_modus = "dot"

# build networks, criterion, optimizers, dataset and dataloader
encoder2 = EncoderRNN2(input_features, hidden_size, n_layers=n_layers, batch_size=batch_size)
decoder2 = LuongAttnDecoderRNN(hidden_size, output_size, n_layers=n_layers, attn_model=attn_modus, batch_size=batch_size)
print(encoder2)
print(decoder2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop([
        {"params": encoder2.parameters()},
        {"params": decoder2.parameters(), "lr": 0.0001}
    ], lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.6)
ds = torchaudio.datasets.YESNO(DATADIR, transform=transforms, target_transform=Labeler())
dl = data.DataLoader(ds, batch_size=batch_size)

if use_cuda:
    print("using CUDA: {}".format(use_cuda))
    encoder2 = encoder2.cuda()
    decoder2 = decoder2.cuda()

loss_total = []
# begin training
for epoch in range(epochs):
    scheduler.step()
    print("epoch {}".format(epoch+1))
    running_loss = 0
    loss_epoch = []
    for i, (mb, tgts) in enumerate(dl):
        # set model into train mode and clear gradients
        encoder2.train()
        decoder2.train()
        encoder2.zero_grad()
        decoder2.zero_grad()

        # set inputs and targets
        mb = mb.transpose(2, 1) # [B x N x L] -> [B, L, N]
        if use_cuda:
            mb, tgts = mb.cuda(), tgts.cuda()
        mb, tgts = Variable(mb), Variable(tgts)

        encoder2_hidden = encoder2.initHidden(type(mb.data))
        encoder2_output, encoder2_hidden = encoder2(mb, encoder2_hidden)
        #print(encoder2_output)

        # Prepare input and output variables for decoder
        dec_i = Variable(encoder2_output.data.new([[[0] * hidden_size] * output_length] * batch_size))
        dec_h = encoder2_hidden # Use last (forward) hidden state from encoder
        #print(dec_h.size())

        """
        # Run through decoder one time step at a time
        # collect attentions
        attentions = []
        outputs = []
        dec_i = Variable(torch.FloatTensor([[[0] * hidden_size] * 1]))
        target_seq = Variable(torch.FloatTensor([[[-1] * hidden_size]*8]))
        for t in range(output_length):
            #print("t:", t, dec_i.size())
            dec_o, dec_h, dec_attn = decoder2(
                dec_i, dec_h, encoder2_output
            )
            #print("decoder output", dec_o.size())
            dec_i = target_seq[:,t].unsqueeze(1) # Next input is current target
            outputs += [dec_o]
            attentions += [dec_attn]
        dec_o = torch.cat(outputs, 1)
        dec_attn = torch.cat(attentions, 1)
        """
        # run through decoder in one shot
        dec_o, dec_h, dec_attn = decoder2(dec_i, dec_h, encoder2_output)

        # calculate loss and backprop
        loss = criterion(dec_o.view(-1, output_size), tgts.view(-1))
        running_loss += loss.data[0]
        loss_epoch += [loss.data[0]]
        loss.backward()
        #nn.utils.clip_grad_norm(encoder2.parameters(), 0.05)
        #nn.utils.clip_grad_norm(decoder2.parameters(), 0.05)
        optimizer.step()

        # logging stuff
        if (i % args.log_interval == 0 and i != 0) or epoch == 0:
            print(loss.data[0])
    loss_total += [loss_epoch]
    print((dec_o.max(2)[1].data == tgts.data).float().sum(1) / tgts.size(1))
    print("ave loss of {} at epoch {}".format(running_loss / (i+1), epoch+1))

loss_total = np.array(loss_total)
plt.figure()
plt.plot(loss_total.mean(1))
plt.savefig("pytorch_attention_audio-loss.png")

# Set up figure with colorbar
attn_plot = dec_attn[0, :, :].data
attn_plot = attn_plot.numpy() if not use_cuda else attn_plot.cpu().numpy()
fig = plt.figure(figsize=(20, 6))
ax = fig.add_subplot(111)
cax = ax.matshow(attn_plot, cmap='bone', aspect="auto")
fig.colorbar(cax)
fig.savefig("pytorch_attention_audio-attention.png")

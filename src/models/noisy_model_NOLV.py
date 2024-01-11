import torch
from torch import nn
from collections import OrderedDict

class NoisySpeechModel(nn.Module):
    def __init__(self, vae, args):
        super(NoisySpeechModel, self).__init__()

        self.vae = vae
        self.input_dim = args.x_dim
        self.h_dim = args.noise_dim_rnn 
        self.z_dim = args.z_dim
        self.num_LSTM = args.noise_num_rnn 
        self.num_dense_noise = args.noise_num_dense 
        self.device = args.device

        # Build the network
        self.build()

    def build(self):

        ########## Noise variance model ##########

        # Define LSTM blocks processing the data input sequence
        self.noise_rnn_x = nn.LSTM(self.input_dim, self.h_dim, self.num_LSTM, bidirectional=False)
        self.noise_rnn_z = nn.LSTM(self.z_dim, self.h_dim, self.num_LSTM, bidirectional=False)

        self.dict_noise_dense = OrderedDict()
        for n in range(self.num_dense_noise):
            if n==0:
                tmp_input_dim = self.h_dim + self.h_dim
                self.dict_noise_dense['linear'+str(n)] = nn.Linear(tmp_input_dim,
                                    self.h_dim)
            else:
                self.dict_noise_dense['linear'+str(n)] = nn.Linear(self.h_dim,
                                    self.h_dim)
            self.dict_noise_dense['tanh'+str(n)] = nn.Tanh()
        self.noise_dense = nn.Sequential(self.dict_noise_dense)

        # Define the linear layer outputing the log-scale
        self.noise_log_scale = nn.Linear(self.h_dim, self.input_dim)
        
    def encode(self, x):
        z, mean, logvar = self.vae.encode(x)
        return mean, logvar, z

    def noiseNN(self, x, z):
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        h0 = torch.zeros(self.num_LSTM, batch_size, 
                        self.h_dim).to(self.device)
        c0 = torch.zeros(self.num_LSTM, batch_size, 
                        self.h_dim).to(self.device)
        h_x, _ = self.noise_rnn_x(x, (h0, c0))
        h_x_shift = torch.zeros(seq_len, batch_size, self.h_dim).to(self.device)
        h_x_shift[1:] = h_x[0:-1]

        h0 = torch.zeros(self.num_LSTM, batch_size, 
                        self.h_dim).to(self.device)
        c0 = torch.zeros(self.num_LSTM, batch_size, 
                        self.h_dim).to(self.device)
        h_z, _ = self.noise_rnn_z(z, (h0, c0))

        h_xz = torch.cat([h_x_shift, h_z], -1)
        noise = self.noise_dense(h_xz)
        var_n = torch.exp(self.noise_log_scale(noise))

        return var_n

    def decode(self, z):
        return self.vae.decode(z)

    def forward(self, x):
        mean, logvar, z = self.encode(x)
        var_s = self.decode(z)
        var_n = self.noiseNN(x, z)  
        return var_s, mean, logvar, z, var_n

    def sample_enc(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)
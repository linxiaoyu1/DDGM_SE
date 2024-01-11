from collections import OrderedDict
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

class RVAE(nn.Module):
    def __init__(self, args):
        super(RVAE, self).__init__()
        self.args = args
        # Load model parameters
        # General
        self.x_dim = self.args.x_dim
        self.z_dim = self.args.z_dim
        self.activation = self._get_activation_fn(self.args.activation)
        self.dropout_p = self.args.dropout
        # Inference
        self.dense_x_gx = [] if self.args.rvae_dense_x_gx == None else [int(i) for i in
                                                                        self.args.rvae_dense_x_gx.split(',')]
        self.dim_RNN_g_x = self.args.rvae_dim_RNN_g_x
        self.num_RNN_g_x = self.args.rvae_num_RNN_g_x
        self.bidir_g_x = self.args.rvae_bidir_g_x
        self.dense_z_gz = [] if self.args.rvae_dense_z_gz == None else [int(i) for i in
                                                                        self.args.rvae_dense_z_gz.split(',')]
        self.dim_RNN_g_z = self.args.rvae_dim_RNN_g_z
        self.num_RNN_g_z = self.args.rvae_num_RNN_g_z
        self.dense_g_z = [] if self.args.rvae_dense_g_z == None else [int(i) for i in
                                                                        self.args.rvae_dense_g_z.split(',')]
        # Generation
        self.dense_z_h = [] if self.args.rvae_dense_z_h == None else [int(i) for i in
                                                                        self.args.rvae_dense_z_h.split(',')]
        self.dim_RNN_h = self.args.rvae_dim_RNN_h
        self.num_RNN_h = self.args.rvae_num_RNN_h
        self.bidir_h = self.args.rvae_bidir_h
        self.dense_h_x = [] if self.args.rvae_dense_h_x == None else [int(i) for i in
                                                                        self.args.rvae_dense_h_x.split(',')]

        # Build the model
        self.build_model()

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == 'silu':
            return nn.SiLU()
        raise RuntimeError("activation should be relu/gelu/tanh/silu, not {}".format(activation))

    def build_model(self):
        ###############################
        ########### Encoder ###########
        ###############################
        # 1. x_t -> g_x_t
        self.dict_mlp_x_gx = OrderedDict()
        if len(self.dense_x_gx) == 0:
            dim_x_gx = self.x_dim
            self.dict_mlp_x_gx['Identity'] = nn.Identity()
        else:
            dim_x_gx = self.dense_x_gx[-1]
            for i in range(len(self.dense_x_gx)):
                if i == 0:
                    self.dict_mlp_x_gx['linear_%s' % str(i)] = nn.Linear(self.x_dim, self.dense_x_gx[i])
                else:
                    self.dict_mlp_x_gx['linear_%s' % str(i)] = nn.Linear(self.dense_x_gx[i-1], self.dense_x_gx[i])
                self.dict_mlp_x_gx['activation_%s' % str(i)] = self.activation
                self.dict_mlp_x_gx['dropout_%s' % str(i)] = nn.Dropout(p=self.dropout_p)

        self.enc_mlp_x_gx = nn.Sequential(self.dict_mlp_x_gx)
        self.enc_rnn_g_x = nn.LSTM(dim_x_gx, self.dim_RNN_g_x, self.num_RNN_g_x,
                               bidirectional=self.bidir_g_x)        

        # 2. z_tm1 -> g_z_t
        self.dict_mlp_z_gz = OrderedDict()
        if len(self.dense_z_gz) == 0:
            dim_z_gz = self.z_dim
            self.dict_mlp_z_gz['Identity'] = nn.Identity()
        else:
            dim_z_gz = self.dense_z_gz[-1]
            for i in range(len(self.dense_z_gz)):
                if i == 0:
                    self.dict_mlp_z_gz['linear_%s' % str(i)] = nn.Linear(self.z_dim, self.dense_z_gz[i])
                else:
                    self.dict_mlp_z_gz['linear_%s' % str(i)] = nn.Linear(self.dense_z_gz[i-1], self.dense_z_gz[i])
                self.dict_mlp_z_gz['activation_%s' % str(i)] = self.activation
                self.dict_mlp_z_gz['dropout_%s' % str(i)] = nn.Dropout(p=self.dropout_p)

        self.enc_mlp_z_gz = nn.Sequential(self.dict_mlp_z_gz)
        self.enc_rnn_g_z = nn.LSTM(dim_z_gz, self.dim_RNN_g_z, self.num_RNN_g_z)
        
        # 3. g_x_t and g_z_t -> z
        num_dir_x = 2 if self.bidir_g_x else 1
        self.dict_mlp_g_z = OrderedDict()
        if len(self.dense_g_z)==0:
            dim_g_z = self.dim_RNN_g_z + num_dir_x * self.dim_RNN_g_x
            self.dict_mlp_g_z['Identity'] = nn.Identity()
        else:
            dim_g_z = self.dense_g_z[-1]
            for i in range(len(self.dense_g_z)):
                if i == 0:
                    self.dict_mlp_g_z['linear'] = nn.Linear(self.dim_RNN_g_z + num_dir_x * self.dim_RNN_g_x, self.dense_g_z[i])
                else:
                    self.dict_mlp_g_z['linear_%s' % str(i)] = nn.Linear(self.dense_g_z[i-1], self.dense_g_z[i])
                self.dict_mlp_g_z['activation'] = self.activation
                self.dict_mlp_g_z['dropout'] = nn.Dropout(p=self.dropout_p)

        self.enc_mlp_g_z = nn.Sequential(self.dict_mlp_g_z)        
        self.enc_inf_z_mean = nn.Linear(dim_g_z, self.z_dim)
        self.enc_inf_z_logvar = nn.Linear(dim_g_z, self.z_dim)
        
        self.module_encoder_layers = [self.enc_mlp_x_gx, self.enc_rnn_g_x, self.enc_mlp_z_gz, self.enc_rnn_g_z, self.enc_mlp_g_z, self.enc_inf_z_mean, self.enc_inf_z_logvar]        
        
        ###############################
        ########### Decoder ###########
        ###############################
        # 1. z_t -> h_t
        self.dict_mlp_z_h = OrderedDict()
        if len(self.dense_z_h) == 0:
            dim_z_h = self.z_dim
            self.dict_mlp_z_h['Identity'] = nn.Identity()
        else:
            dim_z_h = self.dense_z_h[-1]
            for i in range(len(self.dense_z_h)):
                if i == 0:
                    self.dict_mlp_z_h['linear_%s' % i] = nn.Linear(self.z_dim, self.dense_z_h[i])
                else:
                    self.dict_mlp_z_h['linear_%s' % i] = nn.Linear(self.dense_z_h[i-1], self.dense_z_h[i])
                self.dict_mlp_z_h['activation_%s' % i] = self.activation
                self.dict_mlp_z_h['dropout_%s' % i] = nn.Dropout(p=self.dropout_p)
        self.dec_mlp_z_h = nn.Sequential(self.dict_mlp_z_h)

        # 2. h_t
        self.dec_rnn_h = nn.LSTM(dim_z_h, self.dim_RNN_h, self.num_RNN_h, bidirectional=self.bidir_h)

        # 3. h_t -> x_t
        num_dir_h = 2 if self.bidir_h else 1
        self.dict_mlp_h_x = OrderedDict()
        if len(self.dense_h_x) == 0:
            dim_h_x = num_dir_h * self.dim_RNN_h
            self.dict_mlp_h_x['Identity'] = nn.Identity()
        else:
            dim_h_x = self.dense_h_x[-1]
            for i in range(len(self.dense_h_x)):
                if i == 0:
                    self.dict_mlp_h_x['linear_%s' % i] = nn.Linear(num_dir_h * self.dim_RNN_h, self.dense_h_x[i])
                else:
                    self.dict_mlp_h_x['linear_%s' % i] = nn.Linear(self.dense_h_x[i - 1], self.dense_h_x[i])
                self.dict_mlp_h_x['activation_%s' % i] = self.activation
                self.dict_mlp_h_x['dropout_%s' % i] = nn.Dropout(p=self.dropout_p)

        self.dec_mlp_h_x = nn.Sequential(self.dict_mlp_h_x)
        self.dec_gen_x_logvar = nn.Linear(dim_h_x, self.x_dim)
        
        self.module_decoder_layers = [self.dec_mlp_z_h, self.dec_rnn_h, self.dec_mlp_h_x, self.dec_gen_x_logvar]
        

    def encode(self, x):
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # Create variable holder and send to GPU if needed
        z_mean_inf = torch.zeros(seq_len, batch_size, self.z_dim).to(self.args.device)
        z_logvar_inf = torch.zeros(seq_len, batch_size, self.z_dim).to(self.args.device)
        z = torch.zeros(seq_len, batch_size, self.z_dim).to(self.args.device)
        z_t = torch.zeros(batch_size, self.z_dim).to(self.args.device)
        g_z_t = torch.zeros(self.num_RNN_g_z, batch_size, self.dim_RNN_g_z).to(self.args.device)
        c_z_t = torch.zeros(self.num_RNN_g_z, batch_size, self.dim_RNN_g_z).to(self.args.device)

        # 1. x_t to g_x_t
        x_gx = self.enc_mlp_x_gx(x)
        g_x_inverse, _ = self.enc_rnn_g_x(torch.flip(x_gx, [0]))
        g_x = torch.flip(g_x_inverse, [0])

        # 2. z_t to g_z_t, g_x_t and g_z_t to z
        for t in range(seq_len):
            z_gz = self.enc_mlp_z_gz(z_t).unsqueeze(0)
            _, (g_z_t, c_z_t) = self.enc_rnn_g_z(z_gz, (g_z_t, c_z_t))
            g_z_t_last = g_z_t.view(self.num_RNN_g_z, 1, batch_size, self.dim_RNN_g_z)[-1,:,:,:]
            g_z_t_last = g_z_t_last.view(batch_size, self.dim_RNN_g_z)
            concat_xz = torch.cat([g_x[t, :,:], g_z_t_last], -1)
            g_z = self.enc_mlp_g_z(concat_xz)
            z_mean_inf_t = self.enc_inf_z_mean(g_z)
            z_logvar_inf_t = self.enc_inf_z_logvar(g_z)
            z_t = self.reparameterization(z_mean_inf_t, z_logvar_inf_t, sample_mode='logvar')
            z_mean_inf[t, :, :] = z_mean_inf_t
            z_logvar_inf[t, :, :] = z_logvar_inf_t
            z[t,:,:] = z_t

        return z, z_mean_inf, z_logvar_inf

    
    def decode(self, z):
        # 1. z_t to z_h_t
        z_h = self.dec_mlp_z_h(z)
        
        # 2. z_h_t to h_t
        h, _ = self.dec_rnn_h(z_h)
        
        # 3. h_t to y_t
        hx = self.dec_mlp_h_x(h)
        x_gen_logvar = self.dec_gen_x_logvar(hx)
            
        return x_gen_logvar.exp()


    def forward(self, batch):
        x = batch['x']
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        x_dim = x.shape[2]
        assert (x_dim == self.x_dim)
        z, z_mean_inf, z_logvar_inf = self.encode(x)
        x_gen_logvar = self.decode(z)
        
        z_mean_prior = torch.zeros_like(z_mean_inf)
        z_logvar_prior = torch.zeros_like(z_logvar_inf)

        batch['y'] = x_gen_logvar
        batch['z'] = z
        batch['z_mean'] = z_mean_inf
        batch['z_logvar'] = z_logvar_inf
        batch['z_mean_p'] = z_mean_prior
        batch['z_logvar_p'] = z_logvar_prior

        return batch

    def generation_x(self, z):
        # 1. z_t to z_h_t
        z_h = self.dec_mlp_z_h(z)
        
        # 2. z_h_t to h_t
        h, _ = self.dec_rnn_h(z_h)
        
        # 3. h_t to y_t
        hx = self.dec_mlp_h_x(h)
        x_gen_logvar = self.dec_gen_x_logvar(hx)
            
        return x_gen_logvar.exp()

    def reparameterization(self, mean, var, sample_mode):
        if sample_mode == 'logvar':
            std = torch.exp(0.5*var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)
        elif sample_mode == 'var':
            std = torch.sqrt(var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)
        elif sample_mode == 'covar':
            batch_size = mean.shape[0]
            x_dim = mean.shape[1]
            sampled_data = torch.zeros(batch_size, x_dim).to(self.args.device)
            for i in range(batch_size):
                m = MultivariateNormal(mean[i], var[i])
                sampled_data[i, :] = m.sample()
            return sampled_data
        elif sample_mode == 'complex':
            mean_real = mean.real
            mean_imag = mean.imag
            eps_real = torch.rand_like(mean_real)
            eps_imag = torch.rand_like(mean_imag)
            std_real = torch.sqrt(0.5 * var)
            std_imag = torch.sqrt(0.5 * var)
            
            sampled_real = eps_real.mul(std_real).add_(mean_real)
            sampled_imag = eps_imag.mul(std_imag).add_(mean_imag)
            return sampled_real + 1j * sampled_imag    

    def get_info(self):
        info = []
        info.append('========== MODEL INFO ==========')
        info.append('----------- Encoder -----------')
        info.append('x_t to g_x_t:')
        for k, v in self.dict_mlp_x_gx.items():
            info.append('%s : %s' % (k, str(v)))
        info.append(str(self.enc_rnn_g_x))
        info.append('z_tm1 to g_z_t:')
        for k, v in self.dict_mlp_z_gz.items():
            info.append('%s : %s' % (k, str(v)))
        info.append(str(self.enc_rnn_g_z))
        info.append('g_x_t and g_z_t to z_t:')
        for k, v in self.dict_mlp_g_z.items():
            info.append('%s : %s' % (k, str(v)))
        info.append('inf z mean: ' + str(self.enc_inf_z_mean))
        info.append('inf z logvar: ' + str(self.enc_inf_z_logvar))

        info.append('----------- Decoder -----------')
        info.append('z_t to h_t:')
        for k, v in self.dict_mlp_z_h.items():
            info.append('%s : %s' % (k, str(v)))
        info.append(str(self.dec_rnn_h))
        info.append('h_t to x_t:')
        for k, v in self.dict_mlp_h_x.items():
            info.append('%s : %s' % (k, str(v)))
        info.append('gen x: ' + str(self.dec_gen_x_logvar))
        info.append('\n')

        return info











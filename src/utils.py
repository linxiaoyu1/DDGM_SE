import logging
import os
import sys
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .models.rvae import RVAE
from torch.optim import lr_scheduler
from .models import noisy_model_LV, noisy_model_NO, noisy_model_NOLV

def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path, mode='a', encoding='UTF-8')
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

def load_model(args):
    vae = RVAE(args).to(args.device)
    ckpt = torch.load(args.saved_dvae_path, map_location=args.device)
    vae.load_state_dict(ckpt['net_state'])

    # Fix the decoder
    for name in vae.named_parameters():
        if 'mlp_z_h' in name[0]:
            name[1].requires_grad = False
        if 'rnn_h' in name[0]:
            name[1].requires_grad = False
        if 'mlp_h_x' in name[0]:
            name[1].requires_grad = False
        if 'gen_x_logvar' in name[0]:
            name[1].requires_grad = False

    for name in vae.named_parameters():
        if 'dec_' in name[0]:
            name[1].requires_grad = False

    if args.noisy_model == 'LV':
        model = noisy_model_LV.NoisySpeechModel(vae, args).to(args.device)
    elif args.noisy_model == 'NO':
        model = noisy_model_NO.NoisySpeechModel(vae, args).to(args.device)
    elif args.noisy_model == 'NOLV':
        model = noisy_model_NOLV.NoisySpeechModel(vae, args).to(args.device)

    model.train()
    for name in model.named_parameters():
        print(name[0])
        print(name[1].requires_grad)

    return model

def get_tb_writers(args):
    tb_training_dir = os.path.join(args.out_dir, 'tb_log/train')
    tb_val_dir = os.path.join(args.out_dir, 'tb_log/val')
    os.makedirs(tb_training_dir, exist_ok=True)
    os.makedirs(tb_val_dir, exist_ok=True)
    tb_tr_writer = SummaryWriter(tb_training_dir)
    tb_val_writer = SummaryWriter(tb_val_dir)

    return tb_tr_writer, tb_val_writer

def get_scheduler(optimizer, args, total_iter, lr_min):
    if args.policy == 'linear':
        scheduler = lr_scheduler.LinearLR(optimizer, total_iters=args.total_iter) # factor 0.33-1
    elif args.policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter, eta_min=lr_min)
    elif args.policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.decay_step, gamma=0.1)
    elif args.policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=0.05)
    elif args.policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate args.policy [%s] is not implemented', args.policy)
    return scheduler

def loss_function(var_s, mean, logvar, z, var_n, X_abs_2):
    
    eps = 0
    
    batch_size = var_s.shape[1]
    seq_len = var_s.shape[0]
    
    recon = torch.sum( (X_abs_2 + eps)/(var_s + var_n + eps) 
                      - torch.log( (X_abs_2 + eps)/(var_s + var_n + eps) ) 
                      - 1 )
    
    KLD = -0.5 * torch.sum(logvar - mean.pow(2) - logvar.exp())
    
    return recon/(batch_size * seq_len), KLD/(batch_size * seq_len)

def get_stft_dict(args):
    stft_dict = {}
    wlen = args.wlen_sec*args.fs # window length of 64 ms
    wlen = int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
    hop = int(args.hop_percent*wlen) # hop size
    nfft = wlen + args.zp_percent*wlen # number of points of the DFT
    win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

    stft_dict['nfft'] = nfft
    stft_dict['hop'] = hop
    stft_dict['wlen'] = wlen
    stft_dict['win'] = win

    return stft_dict
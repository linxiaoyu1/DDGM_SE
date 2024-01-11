from src.options import get_args_parser
import torch
import os
import json
import numpy as np
from torch import optim
import librosa
from datetime import datetime
from src import speech_dataset
from src.eval import EvalMetrics
from src.utils import get_logger, get_stft_dict, load_model, loss_function

def train_sequence(model, X_abs_2, X, s, args, ind_mix, dataset, logger, stft_dict, save_epoch=100):
    optimizer = optim.Adam(model.parameters(), lr=args.max_lr)
    eval_metrics = EvalMetrics()

    for epoch in range(args.max_epoch):
        optimizer.zero_grad()
        var_s, mean, logvar, z, var_n = model(X_abs_2)
        recon, KLD_z = loss_function(var_s, mean, logvar, z, var_n, X_abs_2)
        loss = recon + args.beta_kld_z*KLD_z
        loss.backward()
        optimizer.step()
        if (epoch % save_epoch == 0) and (epoch != 0) :
            WF = var_s/(var_s + var_n)
            S_recon = (WF).detach().cpu().numpy().squeeze().T*X
            s_recon = librosa.istft(S_recon, hop_length=stft_dict['hop'], win_length=stft_dict['wlen'], window=stft_dict['win'], length=s.shape[0])
            rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi = eval_metrics.eval(x_est=s_recon, x_ref=s, fs=args.fs)
            logger.info("Epoch: {}\t rmse: {:.4f}\t sisdr: {:.2f}\t pypesq: {:.2f}\t pesq wb: {:.2f}\t pesq nb: {:.2f}\t estoi: {:.2f}\t".format(epoch, rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi))
 
            dataset[ind_mix]['epoch{}'.format(epoch)] = {}
            dataset[ind_mix]['epoch{}'.format(epoch)]['rmse'] = rmse
            dataset[ind_mix]['epoch{}'.format(epoch)]['sisdr'] = sisdr
            dataset[ind_mix]['epoch{}'.format(epoch)]['pesq'] = pesq
            dataset[ind_mix]['epoch{}'.format(epoch)]['pesq_wb'] = pesq_wb
            dataset[ind_mix]['epoch{}'.format(epoch)]['pesq_nb'] = pesq_nb
            dataset[ind_mix]['epoch{}'.format(epoch)]['estoi'] = estoi
    
    return dataset

# Get command line arguments
args = get_args_parser()

# Set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Get logger
os.makedirs(args.out_dir, exist_ok=True)
logger = get_logger(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

# Load data info, get STFT settings
with open(args.json_file, 'r') as f:
    dataset = json.load(f)
stft_dict = get_stft_dict(args)

# Training and evaluation
for ind_mix, mix_info in dataset.items():
    utt_name = mix_info['utt_name']
    mix_file = mix_info['noisy_wav'].format(noisy_root=args.noisy_speech_test_dir)
    clean_file = mix_info['clean_wav'].format(clean_root=args.clean_speech_test_dir)
    
    start_time = datetime.now()
    model = load_model(args)
    if args.data_name == 'qut_wsj':
        mix_name, X, X_abs_2, s, x, n = speech_dataset.load_data_wsj(mix_file, args)
    elif args.data_name == 'vb_dmd':
        mix_name, X, X_abs_2, s, x, n = speech_dataset.load_data_vb(mix_file, args)
    mix_name = mix_name.split('.')[0]    
    logger.info('evaluation on sequence {}'.format(mix_name))
    dataset = train_sequence(model, X_abs_2, X, s, args, ind_mix, dataset, logger, stft_dict)

# Write evaluation resuls of audios
json_filename = os.path.join(args.out_dir, 'log_{}.json'.format(args.exp_name))
with open(json_filename, 'w') as f:
    json.dump(dataset, f, indent=1)

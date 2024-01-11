from src.options import get_args_parser
import torch
from torch import optim
import os
import json
import numpy as np
import librosa
from src.utils import get_logger, load_model, get_tb_writers, get_scheduler, loss_function, get_stft_dict
from src import speech_dataset
import copy
from src.eval import eval_dataset

# Get command line arguments
args = get_args_parser()

# Set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Get logger
os.makedirs(args.out_dir, exist_ok=True)
logger = get_logger(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

# Load data, get STFT settings
training_loader, training_size = speech_dataset.build_dataloader(args, data_type='train')
val_loader, val_size = speech_dataset.build_dataloader(args, data_type='val')
val_mix_file_list =  librosa.util.find_files(args.noisy_speech_val_dir, ext='wav')
test_mix_file_list =  librosa.util.find_files(args.noisy_speech_test_dir, ext='wav')
logger.info('Training data size: {}, validation data size: {}'.format(training_size, val_size))

stft_dict = get_stft_dict(args)

# Load model, optimizers, lr scheduler
model = load_model(args)
optimizer = optim.Adam(model.parameters(), lr=args.max_lr)
scheduler = get_scheduler(optimizer, args, args.max_epoch, args.min_lr)

# Get training tb writers
tb_tr_writer, tb_val_writer = get_tb_writers(args)

# Training
logger.info('Start training...')
best_eval_sdr = -np.inf
for epoch in range(args.max_epoch):

    recon_epoch_training = 0
    KLD_z_epoch_training = 0
    loss_epoch_training = 0

    recon_epoch_val = 0
    KLD_z_epoch_val = 0
    loss_epoch_val = 0

    for idx, data in enumerate(training_loader):
        batch_size = data.shape[0]
        optimizer.zero_grad()
        
        data_abs_2 = torch.abs(data)**2
        data_abs_2 = data_abs_2.to(torch.float32).to(args.device).permute(2, 0, 1)

        var_s, mean, logvar, z, var_n = model(data_abs_2)
        recon, KLD_z = loss_function(var_s, mean, logvar, z, var_n, data_abs_2)
        loss = recon + args.beta_kld_z*KLD_z
        loss.backward()
        optimizer.step()

        recon_epoch_training += recon.item() * batch_size
        KLD_z_epoch_training += KLD_z.item() * batch_size
        loss_epoch_training += loss.item() * batch_size

    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()

    recon_tr_avg = recon_epoch_training/training_size
    KLD_z_tr_avg = KLD_z_epoch_training/training_size
    loss_tr_avg = loss_epoch_training/training_size

    logger.info('====> Epoch: {} recon: {:.4f} KLD_z: {:.4f} loss: {:.4f}'.format(
          epoch, recon_tr_avg, KLD_z_tr_avg, loss_tr_avg))
    
    tb_tr_writer.add_scalar('Loss/loss_total', loss_tr_avg, global_step=epoch)
    tb_tr_writer.add_scalar('Loss/loss_recon', recon_tr_avg, global_step=epoch)
    tb_tr_writer.add_scalar('Loss/loss_kld_z', KLD_z_tr_avg, global_step=epoch)
    tb_tr_writer.add_scalar('LR/lr', current_lr, global_step=epoch)

    # Evaluation
    for idx, data in enumerate(val_loader):
        batch_size = data.shape[0]
        
        with torch.no_grad():
            data_abs_2 = torch.abs(data)**2
            data_abs_2 = data_abs_2.to(torch.float32).to(args.device).permute(2, 0, 1)

            var_s, mean, logvar, z, var_n = model(data_abs_2)
            recon, KLD_z = loss_function(var_s, mean, logvar, z, var_n, data_abs_2)
            loss = recon + args.beta_kld_z*KLD_z

            recon_epoch_val += recon.item() * batch_size
            KLD_z_epoch_val += KLD_z.item() * batch_size
            loss_epoch_val += loss.item() * batch_size

    recon_val_avg = recon_epoch_val/val_size
    KLD_z_val_avg = KLD_z_epoch_val/val_size
    loss_val_avg = loss_epoch_val/val_size
    
    logger.info('     Validation: recon: {:.4f} KLD_z: {:.4f} loss: {:.4f}'.format(
          recon_val_avg, KLD_z_val_avg, loss_val_avg))
    
    tb_val_writer.add_scalar('Loss/loss_total', loss_val_avg, global_step=epoch)
    tb_val_writer.add_scalar('Loss/loss_recon', recon_val_avg, global_step=epoch)
    tb_val_writer.add_scalar('Loss/loss_kld_z', KLD_z_val_avg, global_step=epoch)
    
    eval_dict_val = eval_dataset(val_mix_file_list, args.clean_speech_val_dir, model, args, stft_dict)
    if eval_dict_val['si_sdr_avg'] > best_eval_sdr:
        best_eval_sdr = eval_dict_val['si_sdr_avg']
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_best.pt'))
        model_best = copy.deepcopy(model)
        logger.info('Best model at epoch: {} '.format(epoch))
        logger.info('Best SI-SDR on validation set: {}'.format(best_eval_sdr))   

    # Save model
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_latest.pt'))

logger.info('Training finished')
logger.info('Evaluation on complete test set...')
logger.info('test set length: {}'.format(len(test_mix_file_list)))
eval_dict_test = eval_dataset(test_mix_file_list, args.clean_speech_test_dir, model_best, args, stft_dict)

logger.info('Average si-sdr on test: {:.4f} dB\n'.format(eval_dict_test['si_sdr_avg']))
logger.info('Average sdr improvement on test: {:.4f} dB\n'.format(eval_dict_test['si_sdr_improve_avg']))
logger.info('Average rmse on test: {:.4f} dB\n'.format(eval_dict_test['rmse_avg']))
logger.info('Average pesq on test: {:.4f} dB\n'.format(eval_dict_test['pesq_avg']))
logger.info('Average pesq_wb on test: {:.4f} dB\n'.format(eval_dict_test['pesq_wb_avg']))
logger.info('Average pesq_nb on test: {:.4f} dB\n'.format(eval_dict_test['pesq_nb_avg']))
logger.info('Average estoi on test: {:.4f} dB\n'.format(eval_dict_test['estoi_avg']))
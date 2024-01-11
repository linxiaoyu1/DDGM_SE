import numpy as np
import soundfile as sf
from pypesq import pesq as pypesq
from pesq import pesq
from pystoi import stoi
import os
import torch
import librosa
import matplotlib.pyplot as plt
import librosa.display

class EvalMetrics():

    def __init__(self, metric='all'):

        self.metric = metric

    def compute_sisdr(self,s_hat, s_orig):
        """
        Scale Invariant SDR as proposed in
        https://www.merl.com/publications/docs/TR2019-013.pdf
        """
        alpha = s_hat.T @ s_orig / np.sum(np.abs(s_orig)**2)
        SDR = 10*np.log10(np.sum(np.abs(alpha*s_orig)**2)/
                        np.sum(np.abs(alpha*s_orig - s_hat)**2))
        return SDR
    
    def compute_rmse(self,x_est, x_ref):
        # scaling, to get minimum nomrlized-rmse
        alpha = np.sum(x_est*x_ref) / np.sum(x_est**2)
        # x_est_ = np.expand_dims(x_est, axis=1)
        # alpha = np.linalg.lstsq(x_est_, x_ref, rcond=None)[0][0]
        x_est_scaled = alpha * x_est
        return np.sqrt(np.square(x_est_scaled - x_ref).mean())
    
    def compute_median(data):
        median = np.median(data, axis=0)
        q75, q25 = np.quantile(data, [.75 ,.25], axis=0)    
        iqr = q75 - q25
        CI = 1.57*iqr/np.sqrt(data.shape[0])
        if np.any(np.isnan(data)):
            raise NameError('nan in data')
        return median, CI

    def eval(self, x_est, x_ref, fs):

        # mono channel
        if len(x_est.shape) > 1:
            x_est = x_est[:,0]
        if len(x_ref.shape) > 1:
            x_ref = x_ref[:,0]
        # align
        len_x = np.min([len(x_est), len(x_ref)])
        x_est = x_est[:len_x]
        x_ref = x_ref[:len_x]
        if self.metric  == 'rmse':
            return self.compute_rmse(x_est, x_ref)
        elif self.metric == 'sisdr':
            return self.compute_sisdr(x_est, x_ref)
        elif self.metric == 'pesq':
            return pesq(fs, x_ref, x_est, mode='wb'), pesq(fs, x_ref, x_est, mode='nb')
        elif self.metric == 'stoi':
            return stoi(x_ref, x_est, fs, extended=False)
        elif self.metric == 'estoi':
            return stoi(x_ref, x_est, fs, extended=True)
        elif self.metric == 'all':
            score_rmse = self.compute_rmse(x_est, x_ref)
            score_sisdr = self.compute_sisdr(x_est, x_ref)
            score_pesq = pypesq(x_ref, x_est, fs)
            score_pesq_wb = pesq(fs, x_ref, x_est, mode='wb')
            score_pesq_nb = pesq(fs, x_ref, x_est, mode='nb')
            score_estoi = stoi(x_ref, x_est, fs, extended=True)
            return score_rmse, score_sisdr, score_pesq, score_pesq_wb, score_pesq_nb, score_estoi
        else:
            raise ValueError('Evaluation only support: RMSE, SI-SDE, PESQ, (E)STOI, all')
        
def eval_one_sequence(mix_file, speech_dir, epoch, model, args, stft_dict=None, save_examples=False):
    eval_metrics = EvalMetrics()
    
    if args.data_name == 'qut_wsj':
        path, mix_name = os.path.split(mix_file)
        utt_name = mix_name.split('_')[0]
        speech_file = os.path.join(speech_dir, utt_name[:3], utt_name + '.wav')
    elif args.data_name == 'vb_dmd':
        path, mix_name = os.path.split(mix_file)
        speech_file = os.path.join(speech_dir, mix_name)
    
    x, fs_x = sf.read(mix_file) 
    scale = np.max(x)
    x = x/scale

    T_orig = len(x)
    X = librosa.stft(x, n_fft=stft_dict['nfft'], hop_length=stft_dict['hop'], win_length=stft_dict['wlen'], 
                    window=stft_dict['win'])

    if speech_file != None:
        s, fs_s = sf.read(speech_file) 
        s = s/scale
        if len(s) != len(x):
            return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        n = x - s
        S = librosa.stft(s, n_fft=stft_dict['nfft'], hop_length=stft_dict['hop'], win_length=stft_dict['wlen'], 
                    window=stft_dict['win'])
        N_stft = librosa.stft(n, n_fft=stft_dict['nfft'], hop_length=stft_dict['hop'], win_length=stft_dict['wlen'], 
                    window=stft_dict['win'])
        
        input_sdr = eval_metrics.compute_sisdr(x, s)

    F, N = X.shape
    X_abs_2 = np.abs(X)**2

    X_abs_2 = X_abs_2.T
    X_abs_2 = torch.from_numpy(X_abs_2.astype(np.float32))
    X_abs_2 = X_abs_2.to(args.device)
    X_abs_2 = X_abs_2.unsqueeze(1) # (sequence_len, batch_size, input_dim)

    with torch.no_grad():
        var_s, mean, logvar, z, var_n = model(X_abs_2)
        WF = var_s/(var_s + var_n)
        S_recon = (WF).detach().cpu().numpy().squeeze().T*X
        s_recon = librosa.istft(S_recon, n_fft=stft_dict['nfft'], hop_length=stft_dict['hop'], win_length=stft_dict['wlen'], length=x.shape[0])
        rmse, output_sisdr, pesq, pesq_wb, pesq_nb, estoi = eval_metrics.eval(x_est=s_recon, x_ref=s, fs=args.fs)
        sdr_improvement = output_sisdr - input_sdr

    if save_examples:
        save_dir = os.path.join(args.out_dir, 'save_examples')
        save_dir = os.path.join(save_dir, '{}/{}'.format(epoch, mix_name.split('.')[0]))
        os.makedirs(save_dir, exist_ok=True)
        sf.write(os.path.join(save_dir, 'noisy_speech.wav'), x, args.fs)

        if speech_file != None:
            sf.write(os.path.join(save_dir, 'noise.wav'), n, args.fs)
            sf.write(os.path.join(save_dir, 'clean_speech.wav'), s, args.fs)
        
        N_recon = X - S_recon
        n_recon = librosa.istft(N_recon, n_fft=stft_dict['nfft'], hop_length=stft_dict['hop'], win_length=stft_dict['wlen'], length=x.shape[0])

        sf.write(os.path.join(save_dir, 'speech_recon.wav'), s_recon/np.max(np.abs(s_recon)), args.fs)
        sf.write(os.path.join(save_dir, 'noise_recon.wav'), n_recon/np.max(np.abs(n_recon)), args.fs)

        plt.close('all')
        plt.figure()
        librosa.display.specshow(librosa.power_to_db(np.abs(X)**2), 
            y_axis='linear', sr=args.fs, hop_length=stft_dict['hop'], vmin=-40, vmax=40)
        plt.set_cmap('magma')
        plt.colorbar()
        plt.title('noisy speech')
        plt.savefig(os.path.join(save_dir, 'noisy speech.png'))

        if speech_file != None:
            plt.figure()
            librosa.display.specshow(librosa.power_to_db(np.abs(S)**2), 
                y_axis='linear', sr=args.fs, hop_length=stft_dict['hop'], vmin=-40, vmax=40)
            plt.set_cmap('magma')
            plt.colorbar()
            plt.title('clean speech')
            plt.savefig(os.path.join(save_dir, 'clean speech.png'))
            
            plt.figure()
            librosa.display.specshow(librosa.power_to_db(np.abs(N_stft)**2), 
                y_axis='linear', sr=args.fs, hop_length=stft_dict['hop'], vmin=-40, vmax=40)
            plt.set_cmap('magma')
            plt.colorbar()
            plt.title('noise')
            plt.savefig(os.path.join(save_dir, 'noise.png'))

        plt.figure()
        librosa.display.specshow(librosa.power_to_db(var_s.detach().cpu().numpy().squeeze().T), 
            y_axis='linear', sr=args.fs, hop_length=stft_dict['hop'], vmin=-40, vmax=40)
        plt.set_cmap('magma')
        plt.colorbar()
        plt.title('var_s')
        plt.savefig(os.path.join(save_dir, 'estimated speech variance.png'))

        plt.figure()
        plt.imshow(mean)
        plt.set_cmap('magma')
        plt.colorbar()
        plt.title('z_mean')
        plt.savefig(os.path.join(save_dir, 'z_mean.png'))

        plt.figure()
        plt.imshow(logvar)
        plt.set_cmap('magma')
        plt.colorbar()
        plt.title('z_logvar')
        plt.savefig(os.path.join(save_dir, 'z_logvar.png'))

        plt.figure()
        plt.imshow(z)
        plt.set_cmap('magma')
        plt.colorbar()
        plt.title('z')
        plt.savefig(os.path.join(save_dir, 'z.png'))

        plt.figure()
        librosa.display.specshow(librosa.power_to_db(var_n.detach().cpu().numpy().squeeze().T), 
            y_axis='linear', sr=args.fs, hop_length=stft_dict['hop'])
        plt.set_cmap('magma')
        plt.colorbar()
        plt.title('var_n')
        plt.savefig(os.path.join(save_dir, 'estimated noise variance.png'))

        plt.figure()
        librosa.display.specshow(librosa.power_to_db((var_s + var_n).detach().cpu().numpy().squeeze().T), 
            y_axis='linear', sr=args.fs, hop_length=stft_dict['hop'], vmin=-40, vmax=40)
        plt.set_cmap('magma')
        plt.colorbar()
        plt.title('var_s + var_n')
        plt.savefig(os.path.join(save_dir, 'estimated speech + noise variance.png'))

        plt.figure()
        librosa.display.specshow((var_s/(var_s + var_n)).detach().cpu().numpy().squeeze().T, 
            y_axis='linear', sr=args.fs, hop_length=stft_dict['hop'])
        plt.set_cmap('magma')
        plt.colorbar()
        plt.title('Wiener filter')
        plt.savefig(os.path.join(save_dir, 'estimated Wiener filter.png'))

        plt.figure()
        librosa.display.specshow(librosa.power_to_db(np.abs(S_recon)**2), 
            y_axis='linear', sr=args.fs, hop_length=stft_dict['hop'], vmin=-40, vmax=40)
        plt.set_cmap('magma')
        plt.colorbar()
        plt.title('s_from_x')
        plt.savefig(os.path.join(save_dir, 'estimated speech with Wiener filter.png'))

    return sdr_improvement, rmse, output_sisdr, pesq, pesq_wb, pesq_nb, estoi

def eval_dataset(mix_file_list, clean_speech_dir, model, args, stft_dict):
    si_sdr_all = []
    si_sdr_improve_all = []
    rmse_all = []
    pesq_all = []
    pesq_wb_all = []
    pesq_nb_all = []
    estoi_all = []
    for mix_file in mix_file_list:
        path, mix_name = os.path.split(mix_file)
        utt_name = mix_name.split('_')[0]
        epoch = 'None'
        sdr_improvement, rmse, output_sisdr, pesq, pesq_wb, pesq_nb, estoi = eval_one_sequence(mix_file, clean_speech_dir, epoch, model, args, stft_dict=stft_dict, save_examples=False)
        if output_sisdr is not np.nan:
            si_sdr_all.append(output_sisdr)
            si_sdr_improve_all.append(sdr_improvement)
            rmse_all.append(rmse)
            pesq_all.append(pesq)
            pesq_wb_all.append(pesq_wb)
            pesq_nb_all.append(pesq_nb)
            estoi_all.append(estoi)

    eval_dict = {}
    eval_dict['si_sdr_avg'] = np.mean(si_sdr_all)
    eval_dict['si_sdr_improve_avg ']= np.mean(si_sdr_improve_all)
    eval_dict['rmse_avg'] = np.mean(rmse_all)
    eval_dict['pesq_avg'] = np.mean(pesq_all)
    eval_dict['pesq_wb_avg'] = np.mean(pesq_wb_all)
    eval_dict['pesq_nb_avg'] = np.mean(pesq_nb_all)
    eval_dict['estoi_avg'] = np.mean(estoi_all)

    return eval_dict
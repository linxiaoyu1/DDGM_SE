import os
import random
import numpy as np
import soundfile as sf
import librosa
import torch
import pickle
from torch.utils import data

def build_dataloader(args, data_type=None):

    # Load and compute STFT parameters
    stft_dict = {}
    wlen = args.wlen_sec*args.fs # window length of 64 ms
    wlen = int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
    hop = int(args.hop_percent*wlen) # hop size
    nfft = wlen + args.zp_percent*wlen # number of points of the DFT
    win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

    stft_dict['fs'] = args.fs
    stft_dict['nfft'] = nfft
    stft_dict['hop'] = hop
    stft_dict['wlen'] = wlen
    stft_dict['win'] = win
    stft_dict['trim'] = args.trim

    # Generate dataset
    if data_type == 'train':
        file_list = librosa.util.find_files(args.noisy_speech_tr_dir, ext='wav')
    elif data_type == 'val':
        file_list = librosa.util.find_files(args.noisy_speech_val_dir, ext='wav')
    elif data_type == 'test':
        file_list = librosa.util.find_files(args.noisy_speech_test_dir, ext='wav')

    dataset = SpeechSequences(file_list=file_list, sequence_len=args.seq_len,
                                            STFT_dict=stft_dict, shuffle=args.shuffle)
    sample_num = dataset.__len__()

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                                shuffle=args.shuffle, num_workers=args.num_workers)

    return dataloader, sample_num

class SpeechSequences(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    """
    def __init__(self, file_list, sequence_len, STFT_dict, shuffle):

        super().__init__()

        # STFT parameters
        self.fs = STFT_dict['fs']
        self.nfft = STFT_dict['nfft']
        self.hop = STFT_dict['hop']
        self.wlen = STFT_dict['wlen']
        self.win = STFT_dict['win']
        self.trim = STFT_dict['trim']
        
        # data parameters
        self.file_list = file_list
        self.sequence_len = sequence_len
        self.shuffle = shuffle

        self.compute_len()


    def compute_len(self):

        self.valid_seq_list = []

        for wavfile in self.file_list:

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')
        
            if self.trim:
                _, (ind_beg, ind_end) = librosa.effects.trim(x, top_db=30)
            else:
                ind_beg = 0
                ind_end = len(x)

            # Check valid wav files
            seq_length = (self.sequence_len - 1) * self.hop
            file_length = ind_end - ind_beg 
            n_seq = (1 + int(file_length / self.hop)) // self.sequence_len
            for i in range(n_seq):
                seq_start = i * seq_length + ind_beg
                seq_end = (i + 1) * seq_length + ind_beg
                seq_info = (wavfile, seq_start, seq_end)
                self.valid_seq_list.append(seq_info)

        if self.shuffle:
            random.shuffle(self.valid_seq_list)


    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return len(self.valid_seq_list)


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """
        
        # Read wav files
        wavfile, seq_start, seq_end = self.valid_seq_list[index]
        x, fs_x = sf.read(wavfile)

        # Sequence tailor
        x = x[seq_start:seq_end]

        # # Normalize sequence
        x = x/np.max(np.abs(x))

        # STFT transformation
        audio_spec = librosa.stft(x, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen, window=self.win)

        return torch.from_numpy(audio_spec)
    
def load_data_wsj(mix_file, args):
    # STFT parameters
    wlen = args.wlen_sec*args.fs # window length of 64 ms
    wlen = int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
    hop = int(args.hop_percent*wlen) # hop size
    nfft = wlen + args.zp_percent*wlen # number of points of the DFT
    win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

    # speech file corresponding to mixture
    path, mix_name = os.path.split(mix_file)
    utt_name = mix_name.split('_')[0]
    speech_file = os.path.join(args.clean_speech_test_dir, utt_name[:3], utt_name + '.wav')

    # Load mixture
    x, fs_x = sf.read(mix_file) 
    scale = np.max(x)
    x = x/scale
    T_orig = len(x)
    X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, 
                    window=win)
    
    # Load clean
    if speech_file != None:
        s, fs_s = sf.read(speech_file) 
        s = s/scale
        n = x - s
        S = librosa.stft(s, n_fft=nfft, hop_length=hop, win_length=wlen, 
                    window=win)
        N_stft = librosa.stft(n, n_fft=nfft, hop_length=hop, win_length=wlen, 
                    window=win)
    
    F, N = X.shape
    X_abs_2 = np.abs(X)**2

    X_abs_2 = X_abs_2.T
    X_abs_2 = torch.from_numpy(X_abs_2.astype(np.float32))
    X_abs_2 = X_abs_2.to(args.device)
    X_abs_2 = X_abs_2.unsqueeze(1) # (sequence_len, batch_size, input_dim)
    
    return mix_name, X, X_abs_2, s, x, n

def load_data_vb(mix_file, args):
    # STFT parameters
    wlen = args.wlen_sec*args.fs # window length of 64 ms
    wlen = int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
    hop = int(args.hop_percent*wlen) # hop size
    nfft = wlen + args.zp_percent*wlen # number of points of the DFT
    win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

    # speech file corresponding to mixture
    path, mix_name = os.path.split(mix_file)
    speech_file = os.path.join(args.clean_speech_test_dir, mix_name)

    # Load mixture
    x, fs_x = sf.read(mix_file) 
    scale = np.max(x)
    x = x/scale
    T_orig = len(x)
    X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, 
                    window=win)
    
    # Load clean
    if speech_file != None:
        s, fs_s = sf.read(speech_file) 
        s = s/scale
        n = x - s
        S = librosa.stft(s, n_fft=nfft, hop_length=hop, win_length=wlen, 
                    window=win)
        N_stft = librosa.stft(n, n_fft=nfft, hop_length=hop, win_length=wlen, 
                    window=win)
    
    F, N = X.shape
    X_abs_2 = np.abs(X)**2

    X_abs_2 = X_abs_2.T
    X_abs_2 = torch.from_numpy(X_abs_2.astype(np.float32))
    X_abs_2 = X_abs_2.to(args.device)
    X_abs_2 = X_abs_2.unsqueeze(1) # (sequence_len, batch_size, input_dim)
    
    return mix_name, X, X_abs_2, s, x, n


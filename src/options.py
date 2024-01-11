import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Wiener Filter',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Dataset
    parser.add_argument('--data_name', type=str, default='vb_dmd', choices=['qut_wsj', 'vb_dmd'], help='data set name')
    parser.add_argument('--seq_len', type=int, default=100, help='split sequence length (STFT frames)')
    parser.add_argument('--shuffle', type=bool, default=True, help='whether to shuffle the dataset')
    parser.add_argument('--trim', type=bool, default=True, help='whether to trim the speech sequences')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size for training')
    parser.add_argument('--num-workers', type=int, default=6, help='number of thread for loading data')

    ## General
    parser.add_argument('--exp_name', type=str, default=None, help='experience name')
    parser.add_argument('--json_file', type=str, default=None, help='json file for test on dataset')
    parser.add_argument('--x-dim', type=int, default=513, help='dimension of input feature')
    parser.add_argument('--z-dim', type=int, default=16, help='dimension of latent feature z')
    parser.add_argument('--beta-kld-s', type=float, default=1, help='beta for kld s term')
    parser.add_argument('--beta-kld-z', type=float, default=1, help='beta for kld z term')
    parser.add_argument("--dropout", type=float, default=0, help="dropout ratio")
    parser.add_argument('--activation', type=str, default='tanh', choices=['relu', 'gelu', 'silu', 'tanh'], help='activation function for dense layers')

    ## Model loading/saving
    parser.add_argument('--saved-dvae-path', type=str, default='./pretrained_models/vb_rvae_c_decay_3e-3/net_best.pth', help='saved pretrained dvae model path')
    parser.add_argument('--out-dir', type=str, default='./output/test_noise_model', help='save results dir')
    parser.add_argument('--clean-speech-tr-dir', type=str, default='/scratch/bacchus/xilin/data/VoiceBankDemand/clean_trainset_26spk_wav_16k', help='clean speech dir')
    parser.add_argument('--clean-speech-val-dir', type=str, default='/scratch/bacchus/xilin/data/VoiceBankDemand/clean_valset_2spk_wav_16k', help='clean speech validation dir')
    parser.add_argument('--clean-speech-test-dir', type=str, default='/scratch/bacchus/xilin/data/VoiceBankDemand/clean_testset_wav_16k', help='clean speech test dir')
    parser.add_argument('--noisy-speech-tr-dir', type=str, default='/scratch/bacchus/xilin/data/VoiceBankDemand/noisy_trainset_26spk_wav_16k', help='noisy speech dir')
    parser.add_argument('--noisy-speech-val-dir', type=str, default='/scratch/bacchus/xilin/data/VoiceBankDemand/noisy_valset_2spk_wav_16k', help='noisy speech validation dir')
    parser.add_argument('--noisy-speech-test-dir', type=str, default='/scratch/bacchus/xilin/data/VoiceBankDemand/noisy_testset_wav_16k', help='noisy speech test dir')
    parser.add_argument('--save-frequency', type=int, default=1, help='batch training save frequence (epoch)')
    
    ## STFT params
    parser.add_argument("--wlen-sec", type=float, default=64e-3, help="STFT, window length on ms")
    parser.add_argument("--hop-percent", type=float, default=0.25, help="STFT, hop percentage on window length")
    parser.add_argument("--fs", type=float, default=16000, help="STFT, speech frequence")
    parser.add_argument("--zp-percent", type=float, default=0, help="STFT, zp percentage")

    ## RVAE parameters
    parser.add_argument('--rvae-dense-x-gx', type=str, default=None, help='dense layer for x')
    parser.add_argument('--rvae-dim-RNN-g-x', type=int, default=128, help='latent dimension of RNN for g_x')
    parser.add_argument('--rvae-num-RNN-g-x', type=int, default=1, help='number of RNN layers for g_x')
    parser.add_argument('--rvae-bidir-g-x', type=bool, default=False, help='whether use the bidirectional LSTM for g_x')
    parser.add_argument('--rvae-dense-z-gz', type=str, default=None, help='dense layer for z_gz')
    parser.add_argument('--rvae-dim-RNN-g-z', type=int, default=128, help='latent dimension of RNN for g_z')
    parser.add_argument('--rvae-num-RNN-g-z', type=int, default=1, help='number of RNN layers for g_z')
    parser.add_argument('--rvae-dense-g-z', type=str, default='128', help='dense layer for g_z')
    parser.add_argument('--rvae-dense-z-h', type=str, default=None, help='dense layer for z_h')
    parser.add_argument('--rvae-dim-RNN-h', type=int, default=128, help='latent dimension of RNN for h')
    parser.add_argument('--rvae-num-RNN-h', type=int, default=1, help='number of RNN layers for h')
    parser.add_argument('--rvae-bidir-h', type=bool, default=False, help='whether use the bidirectional LSTM for h')
    parser.add_argument('--rvae-dense-h-x', type=str, default=None, help='dense layer for h_x')


    ## Noisy model parameters
    parser.add_argument('--noisy-model', type=str, default='LV', choices=['LV', 'NO', 'NOLV'], help='noisy model name')
    parser.add_argument('--noise-dim-rnn', type=int, default=128, help='latent dimension of RNN for noise model')
    parser.add_argument('--noise-num-rnn', type=int, default=1, help='number of RNN for noise model')
    parser.add_argument('--noise-num-dense', type=int, default=1, help='number of dense layers for noise model')

    ## Training
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='training device')
    parser.add_argument('--max-epoch', type=int, default=300, help='max number of epochs to run')
    parser.add_argument('--max-lr', type=float, default=5e-4, help='max learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-8, help='min learning rate')
    parser.add_argument('--policy', type=str, default='cosine', choices=['linear', 'cosine', 'step', 'multistep', 'plateau'], help="learning rate schedule policy")
    parser.add_argument('--lr-scheduler', type=int, default=[60000], nargs="+", help="learning rate schedule (iterations), only used for MultiStepLR")
    parser.add_argument('--seed', type=int, default=821, help='seed for initializing training')

    return parser.parse_args()
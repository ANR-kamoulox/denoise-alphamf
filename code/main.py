from modules import denoise, vad
import glob, os
from pathlib import Path
import numpy as np
import itertools
import soundfile

#define directory root
dataroot = Path('/mnt/c/Users/Antoine Liutkus/dev/data/MMAD_new')
outputroot = Path("./output")


# Load oracle VAD
vad_file = dataroot / 'clean' / 'VAD.mat'
from scipy.io import loadmat
vad_true = np.squeeze(loadmat(str(vad_file))['VAD'])


# define the different configurations
rooms = ['room_0','room_250','room_500']
noises = ['restaurant', 'babble', 'train']
snrs = ['m5dB','0dB','5dB','10dB']
positions = ['30','90']

# loop over them all
for room, snr, noise, position in itertools.product(rooms, snrs, noises, positions):
    true_noise_basename = 'n%s_%s.wav'%(position,noise)
    mixture_basename = 'noisy%s_%s.wav'%(position,noise)
    estimate_noise_basename = true_noise_basename
    estimate_speech_basename = 'n%s_%s_sp.wav'%(position,noise)


    true_speech_file = dataroot / 'simulated' / room / 'sources' /  'sp.wav'
    true_noise_file = dataroot / 'simulated' / room / 'sources' /  snr / true_noise_basename
    mixture_file = dataroot / 'simulated' / room / 'noisy' /  snr / mixture_basename

    output_speech_filename = outputroot / 'batch' / room / 'sources' / snr / estimate_speech_basename
    output_noise_filename = outputroot / 'batch' / room / 'sources' / snr / estimate_noise_basename
    output_real_filename = outputroot / 'batch' / room / 'noisy' / snr / mixture_basename

    output_speech_filename.parents[0].mkdir(parents = True, exist_ok = True)
    output_real_filename.parents[0].mkdir(parents = True, exist_ok = True)


    (true_speech, fs) = soundfile.read(str(true_speech_file))
    (true_noise, fs) = soundfile.read(str(true_noise_file))
    (mixture, fs) = soundfile.read(str(mixture_file))

    #window = slice(30*fs,60*fs)
    window = slice(0,mixture.shape[0])
    vad_est = vad.vad(mixture[window,:],fs)
    vad_oracle = vad_true[window]

    print(str(mixture_file))
    (real_speech, speech_filtered, noise_filtered) = \
     denoise.alpha_denoise(mixture[window,:], L = 50, alpha = 1.9,
                           nmh = 1, burnin = False, niter = 30,
                           vad = vad_oracle, true_speech=true_speech[window,:],
                           true_noise = true_noise[window,:])

    soundfile.write(str(output_real_filename),real_speech,fs)
    soundfile.write(str(output_speech_filename),speech_filtered,fs)
    soundfile.write(str(output_noise_filename),noise_filtered,fs)

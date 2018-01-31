from modules import denoise, wav
import glob, os

datadir = "/mnt/c/Users/Antoine Liutkus/dev/data/MMAD/simulated/room_0/noisy/5dB"
#datadir = "data"
files =  sorted(glob.glob(os.path.join(datadir,'*.wav')))
for filename in files:
    print(filename)
    (sig,fs) = wav.wavread(filename)
    sig = sig[0:60*fs,:]
    separated = denoise.alpha_denoise(sig, 20, 1.5)
    wav.wavwrite(separated,fs,'test.wav')

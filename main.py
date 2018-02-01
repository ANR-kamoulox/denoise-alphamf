from modules import denoise, wav
import glob, os

#datadir = "/mnt/c/Users/Antoine Liutkus/dev/data/MMAD/simulated/room_0/noisy/5dB"
datadir = "data"
files =  sorted(glob.glob(os.path.join(datadir,'*0db.wav')))
for filename in files:
    print(filename)
    (sig,fs) = wav.wavread(filename)
    sig = sig[35*fs:40*fs,:]
    separated = denoise.alpha_denoise(sig, 10, 1.9, os.path.basename(filename))
    wav.wavwrite(separated,fs,'test.wav')

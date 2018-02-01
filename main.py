from modules import denoise, wav
import glob, os

#datadir = "/mnt/c/Users/Antoine Liutkus/dev/data/MMAD/simulated/room_0/noisy/5dB"
datadir = "data"
files =  sorted(glob.glob(os.path.join(datadir,'*0db.wav')))
for filename in files:
    print(filename)
    (sig,fs) = wav.wavread(filename)
    sig = sig[35*fs:40*fs,:]
    separated = denoise.alpha_denoise(sig, L = 10, alpha = 1.9, sigma = 1e-4,
                                      nmh = 10, burnin = False, name = os.path.basename(filename))
    wav.wavwrite(separated,fs,'test.wav')

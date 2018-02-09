import numpy as np
import itertools
from scipy.signal import stft, istft
from . import stable
from . import beta_ntf
from numpy.linalg import norm



def alpha_denoiseMono(sig, L, alpha,name):

    """Denoising with the mono alpha-stable + NMF model.
   sig is N X K
   L is the number of components """



    # to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # parameters
    nfft = 1024
    niter = 40
    nmh = 40
    nmh_burnin = 30

    # compute STFT of Mixture
    N = sig.shape[0]  # remember number of samples for future use
    X = stft(sig.T, nperseg=nfft)[-1]
    
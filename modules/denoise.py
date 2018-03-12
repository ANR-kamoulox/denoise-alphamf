import numpy as np
from scipy.signal import stft, istft
from . import stable
import tqdm


def invert(M, eps):
    """"inverting matrices M (matrices are the two last dimensions).
    This is assuming that these are 2x2 matrices, using the explicit
    inversion formula available in that case."""
    detM = eps + M[..., 0, 0]*M[..., 1, 1] - M[..., 0, 1]*M[..., 1, 0]
    invDet = 1.0/detM
    invM = np.zeros(M.shape, dtype='complex')
    invM[..., 0, 0] = invDet*M[..., 1, 1]
    invM[..., 1, 0] = -invDet*M[..., 1, 0]
    invM[..., 0, 1] = -invDet*M[..., 0, 1]
    invM[..., 1, 1] = invDet*M[..., 0, 0]
    return invM, np.abs(detM)


def alpha_denoise(
    mixture, L, alpha, nmh=30, burnin=False, niter=10,
    vad=None, true_speech=None, true_noise=None
):
    """Denoising with the multichannel alpha-stable + NMF model.
    mixture is N X K
    L is the number of components """

    # to avoid dividing by zero
    eps = 1e-20  # np.finfo(np.float64).eps

    # parameters
    nfft = 1024
    if burnin:
        nmh_burnin = nmh * 3./4.
    else:
        nmh_burnin = 0

    # compute STFT of Mixture
    N = mixture.shape[0]  # remember number of samples for future use
    (frame_times, X) = stft(mixture.T, nperseg=nfft, noverlap=nfft*3/4)[-2:]
    S = stft(true_speech.T, nperseg=nfft, noverlap=nfft*3/4)[-1]
    R = stft(true_noise.T, nperseg=nfft, noverlap=nfft*3/4)[-1]
    X = np.moveaxis(X, 0, 2)
    S = np.moveaxis(S, 0, 2)
    Z = np.moveaxis(R, 0, 2)

    Xconj = X.conj()
    (F, T, K) = X.shape

    # I: impulse variables for noise
    sigmaI = 2.*np.cos(np.pi*alpha/4.)**(2./alpha)
    I = stable.random(alpha/2., 1, 0, sigmaI, (F, T))

    # W,H: NMF parameters
    W = np.random.rand(F, L)+0.5
    H = np.random.rand(L, T)+0.5
    if vad is not None:
        from scipy.interpolate import interp1d
        vad_function = interp1d(range(N), np.squeeze(vad), 'nearest',
                                bounds_error=False, fill_value=(vad[0], vad[-1]))
        vad_values = vad_function(frame_times)
        H *= vad_values[None, :]
        voice_inactive = np.nonzero(1.-vad_values)[0]
        sigma = np.mean(
            np.mean(np.abs(X[:, voice_inactive, :])**(alpha/2.), axis=2),
            axis=1
        )**2

    else:
        voice_active = np.range(N)
        sigma = np.percentile(
            np.mean(np.abs(X)**(alpha/2.), axis=2), 70, axis=1
        )**2

    # R: Spatial Covarianc Matrices for target
    Id = np.eye(K, dtype='complex64')
    R = np.tile(Id[None, ...], (F, 1, 1))
    R0 = R

    # helper function for computing the sources covariances
    def compute_Cs():
        return np.dot(W, H)[..., None, None] * R[:, None, ...]

    def compute_Cn(impulses):
        return sigma[:, None, None, None] * impulses[..., None, None] * Id[None, None, ...]

    def dot(M, C):
        """M and C are ...x K x K"""
        return np.einsum('...ab,...bc->...ac', M, C)

    def CxxC(C):
        Cx = np.squeeze(dot(C, X[..., None]))  # F x T x K
        return np.einsum('fta,ftb->ftab', Cx, Cx.conj())

    Cs = compute_Cs()
    Cx = Cs + compute_Cn(I)
    (invCx, detCx) = invert(Cx, eps)

    for it in tqdm.tqdm(range(niter+1)):
        # 1/ Metropolis Hasting sampling for the impulses
        # utilitary variables for the expectations wrt phi

        O = np.zeros((F, T, K, K), dtype=np.complex128)
        P = np.zeros((F, T, K, K), dtype=np.complex128)
        Cs = compute_Cs()

        # Metropolis Hastings loop
        count = 0
        for i in tqdm.tqdm(range(nmh)):
            # draw new phi
            Inew = stable.random(alpha/2., 1, 0, sigmaI, (F, T))
            (invCx_n, detCx_n) = invert(Cs + compute_Cn(Inew), eps)

            # compute acceptance probability
            a = (eps + detCx)/(eps + detCx_n) * \
                np.squeeze(np.exp(np.real(
                    dot(dot(Xconj[..., None, :], invCx_n), X[..., None])
                    - dot(dot(Xconj[..., None, :], invCx), X[..., None]),
                )))
            a = np.minimum(1., a)

            # pick the elements that are changed
            u = np.random.rand(F, T)
            changed = np.nonzero(u <= a)

            # update the corresponding impulses and covariances inverses and
            # determinants
            I[changed] = Inew[changed]
            invCx[changed] = invCx_n[changed]
            detCx[changed] = detCx_n[changed]
            if i >= nmh_burnin:
                count += 1
                O += invCx
                P += CxxC(invCx)

            O /= float(count)
            P /= float(count)

        # 2/ Compute posterior statistics for target
        G = dot(Cs, O)
        Cy_post = CxxC(G) + Cs - dot(G, Cs)  # (F, T, K, K) Total variance

        # for the last iteration, we don't update model parameters
        if it == niter:
            break

        # update of  NMF Parameters
        def trRM(R, M):
            """ utilitary function to compute the trace of R times M
            R is FxKxK and M is FxTxKxK"""
            res = np.zeros((F, T), dtype=np.complex)
            for k in range(K):
                res += np.sum(R[:, None, k, :] * M[..., k], axis=-1)
            return res

        zp = np.maximum(0, trRM(R, P).real)
        zo = np.maximum(0, trRM(R, O).real)

        # update W
        num = np.dot(zp, H.T)
        denum = np.dot(zo, H.T)
        W *= np.sqrt((eps+num)/(eps+denum))

        # update H
        num = np.dot(W.T, zp)
        denum = np.dot(W.T, zo)
        H *= np.sqrt((eps+num)/(eps+denum))

        R = np.sum(Cy_post, axis=1)/(eps+np.sum(np.dot(W, H), 1)[..., None, None]) + 1e-5*R0

        # update our model for the mix covariance matrix
        Cx = compute_Cs() + compute_Cn(I)
        (invCx, detCx) = invert(Cx, eps)

    # separates to get the image
    Y_mix = np.squeeze(dot(G, X[..., None]))
    print(X.shape, S.shape, R.shape)
    Y_s = np.squeeze(dot(G, S[..., None]))
    Y_z = np.squeeze(dot(G, Z[..., None]))

    def compute_beamformer(C_post_mean):
        # compute the eigenvalues
        (eig_values, eig_vectors) = np.linalg.eig(C_post_mean)
        # The beamformer maximizing energy is the principal eigenvector for
        # each frequency
        U = np.zeros((F, K)).astype(np.complex)
        for f in range(F):
            index = np.argmax(eig_values[f])
            U[f] = eig_vectors[f, :, index]
        return U

    # Speech output
    # -----------------
    # Compute the average total posterior covariance
    Cy_post_mean = np.mean(Cy_post, axis=1)  # (F, K, K)
    U = compute_beamformer(Cy_post_mean)

    # now apply the beamformer
    S_temp = np.sum((U[:, None, :]).conj() * Y_mix, axis=-1)[None, ...]  # (F, T)
    target_in_mix = np.array(istft(S_temp, nperseg=nfft, noverlap=nfft*3/4)[1]).T[:N]
    S_temp = np.sum((U[:, None, :]).conj() * Y_s, axis=-1)[None, ...]  # (F, T)
    target_in_S = np.array(istft(S_temp, nperseg=nfft, noverlap=nfft*3/4)[1]).T[:N]
    S_temp = np.sum((U[:, None, :]).conj() * Y_z, axis=-1)[None, ...]  # (F, T)
    target_in_Z = np.array(istft(S_temp, nperseg=nfft, noverlap=nfft*3/4)[1]).T[:N]

    return target_in_mix, target_in_S, target_in_Z

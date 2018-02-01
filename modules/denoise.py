import numpy as np
import itertools
from scipy.signal import stft, istft
from . import stable
from . import beta_ntf
from numpy.linalg import norm

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

def alpha_denoise(sig, L, alpha,name):
    """Denoising with the multichannel alpha-stable + NMF model.
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
    X = np.moveaxis(X,0,2)

    sigma = 2e-6
    #sigma = np.percentile(np.abs(X), 30)
    print(sigma)

    Xconj = X.conj()
    (F, T, K) = X.shape

    # Allocate variables
    # W,H: NMF parameters, R: Spatial Covarianc Matrices for target
    # I: impulse variables for noise
    W = np.random.rand(F,L)
    H = np.random.rand(L,T)
    Id = np.eye(K,dtype='complex64')
    R = np.tile(Id[None,...],(F,1,1))
    R0 = R
    sigmaI = 2.*np.cos(np.pi*alpha/4.)**(2./alpha)
    #I = np.ones((F,T))
    I = stable.random(alpha/2., 1, 0, sigmaI, (F,T))

    #helper function for computing the sources covariances
    def compute_Cs():
        return np.dot(W,H)[...,None,None] * R[:,None,...]
    def compute_Cn(impulses):
        return sigma * impulses[...,None,None] * Id[None,None,...]
    def dot(M,C):
        """M and C are ...x K x K"""
        return np.einsum('...ab,...bc->...ac',M,C)

    def CxxC(C):
        Cx = np.squeeze(dot(C, X[...,None])) # F x T x K
        return np.einsum('fta,ftb->ftab',Cx,Cx.conj())

    Cs = compute_Cs()
    Cx = Cs + compute_Cn(I)
    (invCx, detCx) = invert(Cx,eps)

    import tqdm
    for it in tqdm.tqdm(range(niter+1)):
        # 1/ Metropolis Hasting sampling for the impulses
        # utilitary variables for the expectations wrt phi
        O = np.zeros((F,T,K,K),dtype=np.complex128)
        P = np.zeros((F,T,K,K),dtype=np.complex128)
        Cs = compute_Cs()

        #import ipdb; ipdb.set_trace()
        count = 0
        for i in tqdm.tqdm(range(nmh)):
            # draw new phi
            Inew = stable.random(alpha/2., 1, 0, sigmaI, (F,T))
            (invCx_n,detCx_n) = invert(Cs + compute_Cn(Inew),eps)

            #compute acceptance probability
            a = (eps + detCx)/(eps + detCx_n) * \
                np.squeeze(np.exp( np.real(
                     dot(dot(Xconj[...,None,:],invCx_n),X[...,None])
                   - dot(dot(Xconj[...,None,:],invCx),X[...,None]),
                      )))
            #import ipdb; ipdb.set_trace()

            a = np.minimum(1.,a)
            u = np.random.rand(F,T)
            changed = np.nonzero(u<=a)
            #print(float(len(changed[0]))/u.size)

            I[changed]=Inew[changed]
            invCx[changed] = invCx_n[changed]
            detCx[changed] = detCx_n[changed]
            if i >= nmh_burnin:
                count += 1
                O += invCx
                P += CxxC(invCx)

        O /= float(count)
        P /= float(count)

        # for the last iteration, we just need the MCMC part
        if it == niter:
            break

        # now separate sources
        G = np.zeros((F,T,K,K), dtype='complex64')
        Cs = compute_Cs()
        for (i1, i2, i3) in itertools.product(range(K), range(K), range(K)):
                G[..., i1, i2] += Cs[..., i1, i3]*O[..., i3, i2]

        # separates by (matrix-)multiplying this gain with the mix.
        Ys = 0
        for k in range(K):
            Ys += G[..., k]*X[..., k, None]


        # inverte to time domain and return signal
        Ys = np.rollaxis(Ys, -1)  # gets channels back in first position
        target_estimate = istft(Ys)[1].T[:N, :]

        # Stereo to mono stuff
        Varxphi = CxxC(G) + Cs - dot(G,Cs) #(F, T, K, K) Total variation
        Varxphi = np.mean(Varxphi, axis=1) #(F, K, K)

        (eig_values, eig_vectors) = np.linalg.eig(Varxphi)
        idx = eig_values.argsort(axis = -1)[:,::-1]
        U = np.zeros((F, K)).astype(np.complex64) #(F,K) which contains principal eigenvectors
        for f in range(F):
            index = np.argmax(eig_values[f])
            U[f] = eig_vectors[f,:,index]

        S = np.sum((U.T).conj()[..., None] * Ys, axis=0)  # (F, T)
        speech_estimate = np.array(istft(S)[1])[:N]

        from . import wav
        wav.wavwrite(target_estimate,16000,"denoise"+name,verbose=False)
        wav.wavwrite(speech_estimate[:,None], 16000, "speech"+name, verbose=False)

        # 2/ update of  NMF Parameters
        def trRM(R,M):
            """ utilitary function to compute the trace of R times M
            R is FxKxK and M is FxTxKxK"""
            res = np.zeros((F,T),dtype = np.complex128)
            for k in range(K):
                res += np.sum(R[:,None,k,:] * M[...,k], axis = -1)
            return res

        zp = np.maximum(eps, trRM(R,P).real)
        zo = np.maximum(eps, trRM(R,O).real)
        #print('\n',norm(zp),norm(zo),'normes zp zo')

        #update W
        num = np.dot(zp,H.T)
        denum = np.dot(zo,H.T)
        W = W *  np.sqrt((eps+num)/(eps+denum))
        #print('\n',norm(num),norm(denum),'normes num denum W')

        #update H
        num = np.dot(W.T,zp)
        denum = np.dot(W.T,zo)
        H *= np.sqrt((eps+num)/(eps+denum))
        #print('\n',norm(num),norm(denum),'normes num denum H')

        # update our model for the mix covariance matrix
        Cx = compute_Cs() + compute_Cn(I)
        (invCx, detCx) = invert(Cx,eps)


        #update the R
        v = np.dot(W,H)[...,None,None] #spectrogram model F x T x 1 x 1
        A = np.sum(v*O,axis = 1) # FxKxK
        B = dot( dot(R, np.sum(v*P,axis=1)), R)
        zeros = np.zeros((F,K,K))
        block_matrix = np.concatenate(
            (np.concatenate((zeros, -B),axis = 1),
            np.concatenate((-A, zeros),axis = 1)), axis = 2)
        (eig_values,eig_vectors) = np.linalg.eig(block_matrix)

        UH = np.zeros((F,K,K),dtype = np.complex128)
        UG = np.zeros((F,K,K),dtype = np.complex128)
        for f in range(F):
            indices = np.nonzero(eig_values[f] <= 0)
            if not len(indices[0]):
                import ipdb; ipdb.set_trace()
            UH[f,...] = eig_vectors[f,:K,indices[0][:K]]
            UG[f,...] = eig_vectors[f,K:,indices[0][:K]]

        R = dot(UG, invert(UH,eps)[0])+1e-5*R0

        #R = 0.5*(R+np.einsum('fab->fba',R.conj()))
        #R /= np.trace(R,axis1=1,axis2=2)[...,None,None]

        # Update our model for the mix covariance matrix
        Cx = compute_Cs() + compute_Cn(I)
        (invCx, detCx) = invert(Cx,eps)


    return target_estimate

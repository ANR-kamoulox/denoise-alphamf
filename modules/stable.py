# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 2016

@author: Mathieu FONTAINE, mail: mathieu.fontaine@inria.fr

"""
import numpy as np


def random(alpha, beta, mu, sigma, shape, seed = None):
    """
       Input
       -----
       alpha: 0 < alpha <=2
           exponential characteristic coefficient
       beta: -1 <= beta <= 1
           skewness parameter
       mu: real
           the mean
       sigma: positive real
              scale parameter
       shape: as you want :) (give a tuple)
              size and number of sampling

       Returns
       -------
       S: shape
           give a sampling of an S(alpha, beta, mu, sigma) variable

       """
    if seed is None:
        W = np.random.exponential(1, shape)
        U = np.random.uniform(-np.pi / 2., np.pi / 2., shape)

        c = -beta * np.tan(np.pi * alpha / 2.)
        if beta != 1:
            ksi = 1 / alpha * np.arctan(-c)
        else:
            ksi = np.pi / 2.

        res = ((1. + c ** 2) ** (1. / 2. / alpha)) * np.sin(alpha * (U + ksi)) / ((np.cos(U)) ** (1. / alpha)) * ((
                                                                                                                  np.cos(
                                                                                                                      U - alpha * (
                                                                                                                      U + ksi))) / W) ** (
                                                                                                                 (
                                                                                                                 1. - alpha) / alpha)
    else:
        _random = np.random.RandomState(seed)
        W = _random.exponential(1, shape)
        U = _random.uniform(-np.pi / 2., np.pi / 2., shape)

        c = -beta * np.tan(np.pi * alpha / 2.)
        if beta != 1:
            ksi = 1 / alpha * np.arctan(-c)
        else:
            ksi = np.pi / 2.

        res = ((1. + c ** 2) ** (1. / 2. / alpha)) * np.sin(alpha * (U + ksi)) / ((np.cos(U)) ** (1. / alpha)) * ((
                                                                                                                      np.cos(
                                                                                                                          U - alpha * (
                                                                                                                              U + ksi))) / W) ** (
                                                                                                                     (
                                                                                                                         1. - alpha) / alpha)

    return res * sigma + mu

def random_complex_isotropic(alpha=1.2, sigma = 1, shape=()):
    """
        Input
        -----
        alpha: 1e-20 < alpha <=1.9999
            exponential characteristic coefficient
        sigma: positive real
               scale parameter
        shape: as you want :) (give a tuple)
               size and number of sampling

        Returns
        -------
        S: shape
            give a sampling of an isotropic complex variable SalphaS_c(sigma)

        """
    beta = 1
    sigma_imp = 2 * np.cos(np.pi * alpha / 4.) ** float(2. / alpha)  # scale parameter
    imp = random_stable(alpha / 2., beta, 0, sigma_imp, shape) #impulse variable

    # Complex Gaussian variable (important to declare each variable independantly)

    sr = np.random.randn(*shape) * np.sqrt(np.abs(imp)) * sigma * np.sqrt(0.5)  # real part
    si = np.random.randn(*shape) * np.sqrt(np.abs(imp)) * sigma * np.sqrt(0.5)  # imaginary part
    S = sr + 1j * si  # that's our sample of isotropic stable random variable

    return S, imp

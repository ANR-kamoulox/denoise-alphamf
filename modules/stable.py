# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 2016

@author: Mathieu FONTAINE, mail: mathieu.fontaine@inria.fr

"""
import numpy as np


def random(alpha, beta, mu, sigma, shape, seed=None):
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
        if alpha != 1:
            ksi = 1 / alpha * np.arctan(-c)
            res = ((1. + c ** 2) ** (1. / 2. * alpha)) * np.sin(alpha * (U + ksi)) / ((np.cos(U)) ** (1. / alpha)) \
                * ((np.cos(U - alpha * (U + ksi))) / W) ** ((1. - alpha) / alpha)

        else:
            ksi = np.pi / 2.
            res = (1. / ksi) * ((np.pi / 2. + beta * U) * np.tan(U) -
                                beta * np.log((np.pi / 2. * W * np.cos(U)) / (np.pi / 2. + beta * U)))

    else:
        _random = np.random.RandomState(seed)
        W = _random.exponential(1, shape)
        U = _random.uniform(-np.pi / 2., np.pi / 2., shape)

        c = -beta * np.tan(np.pi * alpha / 2.)
        if alpha != 1:
            ksi = 1 / alpha * np.arctan(-c)
            res = ((1. + c ** 2) ** (1. / 2. * alpha)) * np.sin(alpha * (U + ksi)) / ((np.cos(U)) ** (1. / alpha)) \
                * ((np.cos(U - alpha * (U + ksi))) / W) ** ((1. - alpha) / alpha)

        else:
            ksi = np.pi / 2.
            res = (1. / ksi) * ((np.pi / 2. + beta * U) * np.tan(U) -
                                beta * np.log((np.pi / 2. * W * np.cos(U)) / (np.pi / 2. + beta * U)))

    return res * sigma + mu

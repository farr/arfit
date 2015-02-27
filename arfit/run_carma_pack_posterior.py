#!/usr/bin/env python

import argparse
import bz2
import arfit.carma_pack_posterior as cpp
import emcee
import numpy as np
import os
import pickle
import plotutils.autocorr as ac
import plotutils.runner as pr
import sys

class LL(object):
    def __init__(self, logpost):
        self.lp = logpost
    def __call__(self, p):
        return self.lp.log_likelihood(p)

class LP(object):
    def __init__(self, lp):
        self.lp = lp
    def __call__(self, p):
        return self.lp.log_prior(p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, help='data file')
    parser.add_argument('--p', required=True, type=int, help='AR(p)')
    parser.add_argument('--q', required=True, type=int, help='MA(q)')
    
    parser.add_argument('--temps', default=8, type=int, help='number of temperatures (default %(default)s)')
    parser.add_argument('--walkers', default=128, type=int, help='number of walkers (default %(default)s)')
    parser.add_argument('--neff', default=128, type=int, help='number of independent ensembles desired (default %(default)s)')

    parser.add_argument('--reset', default=False, action='store_true', help='reset the sampler before continuing sampling')
        
    args = parser.parse_args()
    
    data = np.loadtxt(args.data)
    data = data[np.argsort(data[:,0]), :] # Sort data
    logpost = cpp.Posterior(data[:,0], data[:,1], data[:,2], p=args.p, q=args.q)
    sampler = emcee.PTSampler(args.walkers, logpost.nparams, LL(logpost), LP(logpost), ntemps=args.temps, adaptation_lag=100, adaptation_time=10, Tmax=np.inf)
    runner = pr.PTSamplerRunner(sampler, np.reshape(np.array([logpost.draw_prior() for i in range(args.temps*args.walkers)]), (args.temps, args.walkers, logpost.nparams)))
    
    try:
        runner.load_state('.')
    except:
        print 'WARNING: could not load saved runner state.'
        sys.__stdout__.flush()
        sampler = emcee.PTSampler(args.walkers, logpost.nparams, LL(logpost), LP(logpost), ntemps=args.temps, adaptation_lag=100, adaptation_time=10, Tmax=np.inf)
        runner = pr.PTSamplerRunner(sampler, np.reshape(np.array([logpost.draw_prior() for i in range(args.temps*args.walkers)]), (args.temps, args.walkers, logpost.nparams)))

    if args.reset:
        runner.reset()

    runner.run_to_neff(args.neff, '.', adapt=True)

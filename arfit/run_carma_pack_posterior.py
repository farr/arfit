#!/usr/bin/env python

import argparse
import bz2
import arfit.carma_pack_posterior as cpp
import emcee
import numpy as np
import os
import pickle
import plotutils.autocorr as ac
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
    
    parser.add_argument('--thin', default=100, type=int, help='thin parameter (default %(default)s)')
    parser.add_argument('--save', default=1000, type=int, help='steps between save (default %(default)s)')

    parser.add_argument('--temps', default=8, type=int, help='number of temperatures (default %(default)s)')
    parser.add_argument('--walkers', default=128, type=int, help='number of walkers (default %(default)s)')
    
    args = parser.parse_args()
    
    try:
        with bz2.BZ2File('sampler.pkl.bz2', 'r') as inp:
            sampler = pickle.load(inp)
        result = [sampler.chain[:,:,-1,:]]
    except:
        data = np.loadtxt(args.data)
        logpost = cpp.Posterior(data[:,0], data[:,1], data[:,2], p=args.p, q=args.q)
        sampler = emcee.PTSampler(args.walkers, logpost.nparams, LL(logpost), LP(logpost), ntemps=args.temps, adaptation_lag=100, adaptation_time=10, Tmax=np.inf)
        result = [np.reshape(np.array([logpost.draw_prior() for i in range(args.temps*args.walkers)]), (args.temps, args.walkers, logpost.nparams))]

    while True:
        result = sampler.run_mcmc(result[0], args.save, adapt=True, thin=args.thin)

        with bz2.BZ2File('sampler.pkl.bz2.temp', 'w') as out:
            pickle.dump(sampler, out)
        os.rename('sampler.pkl.bz2.temp', 'sampler.pkl.bz2')

        print 'Saved state after ', sampler.chain.shape[2], ' iterations.'
        print
        print 'ACL is ', ac.emcee_ptchain_autocorrelation_lengths(sampler.chain)
        print
        print 'Ensemble acceptance rate is ', np.mean(sampler.acceptance_fraction, axis=1)
        print
        print 'Temperature acceptance rate is ', sampler.tswap_acceptance_fraction
        print
        print
        sys.__stdout__.flush()

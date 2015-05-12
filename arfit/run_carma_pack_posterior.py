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

    parser.add_argument('--init', default=None, help='initialisation state file')

    parser.add_argument('--data', required=True, help='data file')
    parser.add_argument('--p', required=True, type=int, help='AR(p)')
    parser.add_argument('--q', required=True, type=int, help='MA(q)')
    
    parser.add_argument('--temps', default=8, type=int, help='number of temperatures (default %(default)s)')
    parser.add_argument('--walkers', default=128, type=int, help='number of walkers (default %(default)s)')
    parser.add_argument('--neff', default=128, type=int, help='number of independent ensembles desired (default %(default)s)')

    parser.add_argument('--threads', default=1, type=int, help='number of threads in sampler (default %(default)s)')

    parser.add_argument('--reset', default=False, action='store_true', help='reset the sampler before continuing sampling')
    parser.add_argument('--reset-once', default=False, action='store_true', help='reset the sampler once (remove \'reset.once\' to reset again)')
        
    args = parser.parse_args()
    
    data = np.loadtxt(args.data)

    times, tind = np.unique(data[:,0], return_index=True)
    data = data[tind, :]
    
    logpost = cpp.Posterior(data[:,0], data[:,1], data[:,2], p=args.p, q=args.q)
    sampler = emcee.PTSampler(args.walkers, logpost.nparams, LL(logpost), LP(logpost), ntemps=args.temps, adaptation_lag=100, adaptation_time=10, Tmax=np.inf, threads=args.threads)
    runner = pr.PTSamplerRunner(sampler, np.reshape(np.array([logpost.draw_prior() for i in range(args.temps*args.walkers)]), (args.temps, args.walkers, logpost.nparams)))
    
    try:
        runner.load_state('.')
    except:
        print 'WARNING: could not load saved runner state.'
        sys.__stdout__.flush()

        if args.init is not None:
            with bz2.BZ2File(args.init, 'r') as inp:
                init = np.load(inp)
        else:
            init = np.reshape(np.array([logpost.draw_prior() for i in range(args.temps*args.walkers)]), (args.temps, args.walkers, logpost.nparams))

        sampler = emcee.PTSampler(args.walkers, logpost.nparams, LL(logpost), LP(logpost), ntemps=args.temps, adaptation_lag=100, adaptation_time=10, Tmax=np.inf, threads=args.threads)
        runner = pr.PTSamplerRunner(sampler, init)

    if args.reset:
        runner.reset()
    if args.reset_once and runner.chain is not None:
        try:
            with open('reset.once', 'r') as inp:
                pass
        except:
            # Couldn't open reset.once, so reset
            runner.reset()
            with open('reset.once', 'w') as out:
                out.write('Reset!\n')

    runner.run_to_neff(args.neff, '.', adapt=True)

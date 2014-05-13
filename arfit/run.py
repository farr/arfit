#!/usr/bin/env python

import ar1_posterior as pos
import argparse
import bz2
import emcee
import numpy as np
import os.path as op
import os
import plotutils.autocorr as ac
import sys

def load_data(file):
    data = np.loadtxt(file)

    ts, tinds = np.unique(data[:,0], return_index = True)

    data = data[tinds, :]

    return data
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, help='data file')
    parser.add_argument('--outdir', default='.', help='output directory')
    parser.add_argument('--nwalkers', default=128, type=int, help='number of walkers')
    parser.add_argument('--nensembles', default=128, type=int, help='number of independent ensembles')
    parser.add_argument('--nthreads', default=1, type=int, help='number of threads')
    parser.add_argument('--nthin', default=10, type=int, help='number of steps between saved states')
    parser.add_argument('--nsave', default=1280, type=int, help='number of steps between stored states')

    args = parser.parse_args()

    data = load_data(args.data)

    print 'Loaded data'
    sys.stdout.flush()

    logpost = pos.AR1Posterior(data[:,0], data[:,1])

    sampler = emcee.EnsembleSampler(args.nwalkers, 2, logpost, threads=args.nthreads)

    with bz2.BZ2File(op.join(args.outdir, 'chain.npy.bz2'), 'r') as cinp:
        with bz2.BZ2File(op.join(args.outdir, 'lnprob.npy.bz2'), 'r') as pinp:
            sampler._chain = np.load(cinp)
            sampler._lnprob = np.load(pinp)

    print 'Loaded previous chain file'
    sys.stdout.flush()

    tchain = ac.emcee_thinned_chain(sampler.chain)
    while tchain is None or tchain.shape[1] < args.nensembles:
        sampler.run_mcmc(sampler.chain[:,-1,:], args.nsave, thin=args.nthin)

        with bz2.BZ2File(op.join(args.outdir, 'temp_chain.npy.bz2'), 'w') as out:
            with bz2.BZ2File(op.join(args.outdir, 'temp_lnprob.npy.bz2'), 'w') as pout:
                np.save(out, sampler.chain)
                np.save(pout, sampler.lnprobability)
        os.rename(op.join(args.outdir, 'temp_chain.npy.bz2'),
                  op.join(args.outdir, 'chain.npy.bz2'))
        os.rename(op.join(args.outdir, 'temp_lnprob.npy.bz2'),
                  op.join(args.outdir, 'lnprob.npy.bz2'))

        tchain = ac.emcee_thinned_chain(sampler.chain)

        if tchain is None:
            print 'Saved state after ', sampler.chain.shape[1], ' ensembles, no ACL'
            sys.stdout.flush()
        else:
            print 'Saved state after ', sampler.chain.shape[1], ' ensembles, ', tchain.shape[1], ' independent ensembles'
            sys.stdout.flush()

    with bz2.BZ2File(op.join(args.outdir, 'tchain.npy.bz2'), 'w') as out:
        np.save(out, tchain)
        
        
        

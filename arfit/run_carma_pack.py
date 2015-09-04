#!/usr/bin/env python

from __future__ import print_function

import argparse
import carmcmc as cm
import numpy as np
import os
import plotutils.autocorr as ac
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, metavar='FILE', help='input file')
    parser.add_argument('--output', required=True, metavar='FILE', help='chain output')

    parser.add_argument('--p', default=3, type=int, metavar='P', help='AR order (default: %(default)s)')
    parser.add_argument('--q', default=2, type=int, metavar='Q', help='MA order (default: %(default)s)')

    parser.add_argument('--neff', default=1000, type=int, metavar='N', help='number of independent samples (default: %(default)s)')

    args = parser.parse_args()
    
    data = np.loadtxt(args.input)

    times, tind = np.unique(data[:,0], return_index=True)
    data = data[tind, :]
    
    model = cm.CarmaModel(data[:,0], data[:,1], data[:,2], p=args.p, q=args.q)

    thin = 1
    nsamp = 10*args.neff

    out, ext = os.path.splitext(args.output)
    outtemp = out + '.TEMP' + ext
    
    while True:
        sample = model.run_mcmc(nsamp, nthin=thin, nburnin=thin*nsamp/2)

        np.savetxt(outtemp, np.column_stack((sample.trace, sample.get_samples('loglik'), sample.get_samples('logpost'))))
        os.rename(outtemp, args.output)

        taus = []
        for j in range(sample.trace.shape[1]):
            taus.append(ac.autocorrelation_length_estimate(sample.trace[:,j]))
        taus = np.array(taus)

        if np.any(np.isnan(taus)):
            neff_achieved = 0
        else:
            neff_achieved = sample.trace.shape[0] / np.max(taus)

        print('Ran for ', nsamp*thin, ' steps, achieved ', neff_achieved, ' independent samples')
        sys.__stdout__.flush()
            
        if neff_achieved >= args.neff:
            break
        else:
            thin *= 2

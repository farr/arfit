import carmcmc as cm
import numpy as np

def csample_from_files(datafile, chainfile, p, q):
    data = np.loadtxt(datafile)
    chain = np.loadtxt(chainfile)
    assert chain.shape[1] == p + q + 5, 'dimension mismatch'

    return cm.CarmaSample(data[:,0], data[:,1], data[:,2], None, q=q, trace=chain[:,:-2], loglike=chain[:,-2], logpost=chain[:,-1])

import numpy as np

def kalman_prediction_and_variance(ts, ys, dys, mu, sigma, ar_roots, ma_roots):
    """Outputs the prediction of a Kalman filter implementing a CARMA
    model and the associated prediction variance for data ``ys`` taken
    at times ``ts`` with measurement uncertanties ``dys``.

    :param ts: The sample times.

    :param ys: The sample data.

    :param dys: The (1-sigma) measurement uncertainties.

    :param mu: The (constant) mean of the filter process.

    :param sigma: The (constant) standard deviation of the filter
      process.

    :param ar_roots: The (possibly complex) roots describing the
      autoregressive part of the CARMA process.

    :param ma_roots: The (possibly complex) roots describing the
      moving average part of the CARMA process.

    :returns: ``(ys_pred, var_ys_pred)``, a tuple of arrays.
      ``ys_pred[i]`` gives the predicted value of time series element
      ``i`` given all the elements, ``j``, with ``j < i`` from the
      filter; ``var_ys_pred[i]`` gives the variance of this
      prediction, incorporating the quoted observational uncertainty.

    """
    
    ts = np.atleast_1d(ts)
    ys = np.atleast_1d(ys)
    dys = np.atleast_1d(dys)

    ar_roots = np.atleast_1d(ar_roots)
    ma_roots = np.atleast_1d(ma_roots)

    p = ar_roots.shape[0]
    q = ma_roots.shape[0]

    # polyfromroots returns
    # c[0] + c[1]*x + c[2]*x**2 + ... + x**q
    ma_poly = np.polynomial.polynomial.polyfromroots(ma_roots)
    ma_poly = ma_poly / ma_poly[0] # Make c[0] == 1
    ma_poly = np.concatenate((ma_poly,np.zeros(p-q-1))) # Higher order terms are zero.

    # Recentre
    ys = ys - mu

    U = np.zeros((p,p), dtype=np.complex)
    for l in range(p):
        for k in range(p):
            U[l,k] = ar_roots[k]**l

    btilde = np.dot(ma_poly, U)

    e = np.zeros(p)
    e[-1] = sigma # This will be adjusted later....

    J = np.linalg.solve(U, e)

    Vtilde = np.zeros((p,p), dtype=np.complex)
    for l in range(p):
        for k in range(p):
            Vtilde[l,k] = -J[l]*np.conj(J[k])/(ar_roots[l] + np.conj(ar_roots[k]))

    xtilde = np.zeros(p)
    Ptilde = Vtilde.copy()

    # Here we adjust the variance matrix so that the variance of the
    # process is sigma*sigma
    R0 = np.dot(btilde, np.dot(Ptilde, np.conj(btilde)))
    sigma_factor2 = sigma*sigma/R0
    Vtilde *= sigma_factor2
    Ptilde *= sigma_factor2

    eys = [0.0]
    vys = [np.dot(btilde, np.dot(Ptilde, np.conj(btilde))) + dys[0]*dys[0]]

    gain = np.dot(Ptilde, np.conj(btilde)) / vys[0]

    xtilde = xtilde + ys[0]*gain
    Ptilde = Ptilde - vys[0]*np.outer(gain, np.conj(gain))

    eigs = np.linalg.eigvalsh(Ptilde)
    assert np.all(eigs > 0), 'Ptilde has become non-positive-definite'
    
    for i in range(1, ys.shape[0]):
        dt = ts[i] - ts[i-1]
        Lambda = np.diag(np.exp(ar_roots*dt))

        xtilde = np.dot(Lambda, xtilde)
        Ptilde = np.dot(Lambda, np.dot(Ptilde - Vtilde, np.conj(Lambda.T))) + Vtilde

        eigs = np.linalg.eigvalsh(Ptilde)
        assert np.all(eigs > 0), 'Ptilde has become non-positive-definite'

        eys.append(np.dot(btilde, xtilde))
        vys.append(np.dot(btilde, np.dot(Ptilde, np.conj(btilde))) + dys[i]*dys[i])

        gain = np.dot(Ptilde, np.conj(btilde))/vys[i]

        xtilde = xtilde + (ys[i] - eys[i])*gain
        Ptilde = Ptilde - vys[i]*np.outer(gain, np.conj(gain)) 

        eigs = np.linalg.eigvalsh(Ptilde)
        assert np.all(eigs > 0), 'Ptilde has become non-positive-definite'
        
    eys = np.real(np.array(eys))
    vys = np.real(np.array(vys))
        
    return eys + mu, vys

import numpy as np
from multiprocessing import Pool

def reconstruct(x, dim, tau=1):
    m = len(x) - (dim - 1) * tau
    print(m)
    if m <= 0:
        raise ValueError('Length of the time series is <= (dim - 1) * tau.')
    
    ret = np.asarray([x[i:i + (dim - 1) * tau + 1:tau] for i in range(m)])
    delay = ret.reshape(tau*dim,-1).T
    return delay

def acorr(x, maxtau=100, norm=True, detrend=True):
    x = np.asarray(x)
    N = len(x)
    if not maxtau:
        maxtau = N
    else:
        maxtau = min(N, maxtau)
    if detrend:
        x = x - np.mean(x)
    y = np.fft.fft(x, 2 * N - 1)
    r = np.real(np.fft.ifft(y * y.conj(), 2 * N - 1))
    if norm:
        return r[:maxtau] / r[0]
    else:
        return r[:maxtau]

def generate_tau_estimates(X):
    I_VAL =[]
    R_VAL = []
    I = []
    R = []
    print(X.shape)
    for c in range(X.shape[0]):
        print(c)
        x = X[c]
        r = acorr(x, maxtau=100)
        r_delay = np.argmax(r < 1.0 / np.e)
        R_VAL.append(r_delay)

    return R_VAL


def falsify(X, dims=20, window=10):
    dim = np.arange(1, dims)
    F3S = []
    for c in range(X.shape[0]):
        print(c)
        x = X[c]
        f1, f2, f3 = fnn(x, tau=1, dim=dim + 2, window=window)
        F3S.append(f3)
    return F3S


def _fnn(d, x, tau=1, R=10.0, A=2.0, metric='euclidean', window=10,
         maxnum=None):
    # We need to reduce the number of points in dimension d by tau
    # so that after reconstruction, there'll be equal number of points
    # at both dimension d as well as dimension d + 1.
    y1 = utils.reconstruct(x[:-tau], d, tau)
    y2 = utils.reconstruct(x, d + 1, tau)

    # Find near neighbors in dimension d.
    index, dist = utils.neighbors(y1, metric=metric, window=window,
                                  maxnum=maxnum)

    # Find all potential false neighbors using Kennel et al.'s tests.
    f1 = np.abs(y2[:, -1] - y2[index, -1]) / dist > R
    f2 = utils.dist(y2, y2[index], metric=metric) / np.std(x) > A
    f3 = f1 | f2

    return np.mean(f1), np.mean(f2), np.mean(f3)


def fnn(x, dim=[1], tau=1, R=10.0, A=2.0, metric='euclidean', window=10,
        maxnum=None, parallel=True):
    if parallel:
        processes = None
    else:
        processes = 1

    return parallel_map(_fnn, dim, (x,), {
                              'tau': tau,
                              'R': R,
                              'A': A,
                              'metric': metric,
                              'window': window,
                              'maxnum': maxnum
                              }, processes).T

def parallel_map(func, values, args=tuple(), kwargs=dict(),
                 processes=None):
    # True single core processing, in order to allow the func to be executed in
    # a Pool in a calling script.
    if processes == 1:
        return np.asarray([func(value, *args, **kwargs) for value in values])

    pool = Pool(processes=processes)
    results = [pool.apply_async(func, (value,) + args, kwargs)
               for value in values]

    pool.close()
    pool.join()

    return np.asarray([result.get() for result in results])

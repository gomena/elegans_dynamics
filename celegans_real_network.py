import os
import itertools as it
import time
import csv
import autograd.numpy as np
import autograd.numpy.random as npr
import pickle

from autograd.scipy.misc import logsumexp
from autograd.scipy.special import gammaln
from scipy.optimize import linear_sum_assignment
from autograd.util import flatten
from autograd.optimizers import adam, sgd
from autograd import grad

import tensorflow as tf
import sys
sys.path.insert(0, '../')



from copy import deepcopy

### Set the inputs/outputs
CELEGANS_NETWORK = os.path.join("data", "C-elegans-frontal.txt")
CELEGANS_METADATA = os.path.join("data", "C-elegans-frontal-meta.csv")


def cached(results_dir, results_name):
    def _cache(func):

        def func_wrapper(*args, **kwargs):
            results_file = os.path.join(results_dir, results_name)
            if not results_file.endswith(".pkl"):
                results_file += ".pkl"

            print(results_file)
            if os.path.exists(results_file):
                with open(results_file, "rb") as f:
                    results = pickle.load(f)

            else:
                results = func(*args, **kwargs)
                with open(results_file, "wb+") as f:
                    pickle.dump(results, f)

            return results
        return func_wrapper
    return _cache


def gaussian_entropy(log_sigma):
    return 0.5 * log_sigma.size * (1.0 + np.log(2 * np.pi)) + np.sum(log_sigma)


def logistic(psi):
    return 1. / (1 + np.exp(-psi))


def load_celegans_network(props = np.ones((3, 4))):
    """" This function loads a connectome with a subsample of the entire connectome. The sub-sample
        is given by props. props[i,j] = proportion of neurons of category (i,j) to include
        category i = body position (Head = 0, Middle =1, Tail =2)
        category j = neuron type (Sensory = 0, Motor = 1, Interneuron =2, Poly-type =3)
        Besides names and positions of neurons, it outputs an array of adjacency matrix, for each type of
        connectivity (Synapse, electric junction and NMJ (?))"""


    NeuronTypeCSV = csv.reader(open('data/NeuronType.csv', 'r'), delimiter=',', skipinitialspace=True)
    neuron_info_all = [[] for index in range(4)]
    relevant_indexes = [0, 1, 2, 14]
    # load relevant information (names, numerica position, anatomical position and type)
    for row in NeuronTypeCSV:
        for j0, j in enumerate(relevant_indexes):
            neuron_info_all[j0].append(row[j].strip(' \t\n\r'))

    names_with_zeros = deepcopy(neuron_info_all[0])
    # erase extra zeros in name
    for j in range(279):
        indZero = neuron_info_all[0][j].find('0')
        if (indZero >= 0 and indZero < len(neuron_info_all[0][j]) - 1):
            neuron_info_all[0][j] = neuron_info_all[0][j].replace('0', '')

    names = deepcopy(neuron_info_all[0])
    xpos = np.array(neuron_info_all[1])

    location = neuron_info_all[2]

    issensory = np.zeros(279)
    ismotor = np.zeros(279)
    isinterneuron = np.zeros(279)

    NeuronTypeISM = csv.reader(open('data/NeuronTypeISM.csv', 'r'), delimiter=',', skipinitialspace=True)

    for row in NeuronTypeISM:
        try:
            index = names.index(row[0])
            words = row[2].lower()
            if ('sensory' in words):
                issensory[index] = 1
            if ('motor' in words):
                ismotor[index] = 1
            if ('interneuron' in words):
                isinterneuron[index] = 1
        except:
            pass
    NeuronRemainingTypesISM = csv.reader(open('data/NeuronRemainingTypesISM.csv', 'r'), delimiter=',',
                                         skipinitialspace=True)
    for row in NeuronRemainingTypesISM:
        try:
            index = neuron_info_all[0].index(row[0])
            words = row[1].lower()
            if ('sensory' in words):
                issensory[index] = 1
            if ('motor' in words):
                ismotor[index] = 1
            if ('interneuron' in words):
                isinterneuron[index] = 1
        except:
            pass

    ConnectomeCSV = csv.reader(open('data/NeuronConnect.csv', 'r'), delimiter=',', skipinitialspace=True)
    As_weighted = np.zeros((3, 279, 279))

    for row in ConnectomeCSV:
        try:
            index1 = names_with_zeros.index(row[0])
            index2 = names_with_zeros.index(row[1])
            if ('S' in row[2] or 'R' in row[2] or 'Sp' in row[2] or 'Rp' in row[2]):
                As_weighted[0, index1, index2] = As_weighted[0, index1, index2] + float(row[3])
            if ('EJ' in row[2]):
                As_weighted[1, index1, index2] = As_weighted[1, index1, index2] + float(row[3])
            if ('NMJ' in row[2]):
                As_weighted[2, index1, index2] = As_weighted[2, index1, index2] + float(row[3])
        except:
            pass
    As = (As_weighted > 0).astype(int)

    ind_type = [[] for _ in range(4)]

    # 0=sensory,motor,interneuron,poly
    ind_type[0] = np.where(
        np.logical_and(np.logical_and(issensory.astype(bool), (1 - ismotor).astype(bool)),
                       (1 - isinterneuron).astype(bool)))[0]
    ind_type[1] = np.where(
        np.logical_and(np.logical_and((1 - issensory).astype(bool), ismotor.astype(bool)),
                       (1 - isinterneuron).astype(bool)))[0]
    ind_type[2] = np.where(
        np.logical_and(np.logical_and((1 - issensory).astype(bool), (1 - ismotor).astype(bool)),
                       isinterneuron.astype(bool)))[0]

    ind_type[3] = np.where(issensory + ismotor + isinterneuron >= 2)[0]

    # Head, Middle, Tail

    ind_pos = [[] for _ in range(3)]
    ind_pos[0] = [i for i, j in enumerate(location) if j == 'H']
    ind_pos[1] = [i for i, j in enumerate(location) if j == 'M']
    ind_pos[2] = [i for i, j in enumerate(location) if j == 'T']

    ind_type_pos_number = np.zeros((3, 4))

    ind_type_pos = [[] for _ in range(3)]

    for j in range(3):
        ind_type_pos[j] = [[] for _ in range(4)]

    for i in range(4):
        for j in range(3):
            ind_type_pos[j][i] = [val for val in ind_pos[j] if val in ind_type[i]]
            ind_type_pos_number[j, i] = len(ind_type_pos[j][i])

    ind_neuron_subsampled = [[] for _ in range(3) for _ in range(4)]
    for j in range(3):
        ind_neuron_subsampled[j] = [[] for _ in range(4)]

    for i in range(4):
        for j in range(3):
            try:
                ind_neuron_subsampled[j][i] = np.random.choice(ind_type_pos[j][i],
                                                               np.floor(ind_type_pos_number[j, i] * props[j, i]).astype(
                                                                   int), replace=False)
            except:
                ind_neuron_subsampled[j][i] = []

    ind_neuron_subsampled = np.sort(
        np.concatenate([np.concatenate(ind_neuron_subsampled[j][:], axis=0) for j in range(3)]).astype(int))

    As = As[np.ix_(range(3), ind_neuron_subsampled, ind_neuron_subsampled)]
    xpos = np.array(deepcopy(xpos[ind_neuron_subsampled]).astype(float))
    names = [j for j0, j in enumerate(names) if j0 in ind_neuron_subsampled]

    return As, names, xpos


def simulate_celegans(A, posx, M, T, num_given, dthresh=0.01,
                      sigmasq_W=None, etasq=0.1, spectral_factor = 1.0):
    N = A.shape[0]
    rho = np.mean(A.sum(0))

    # Set sigmasq_W for stability
    sigmasq_W = sigmasq_W if sigmasq_W is not None else 1./(1.1 * N * rho)


    W = (npr.randn(N, N) *A)
    W = (W-W.T)/2
    #W =np.identity(N) * A

    eigmax = np.max(abs(np.linalg.eig(W)[0]))


    W = W/ (spectral_factor * eigmax)

    assert np.max(abs(np.linalg.eigvals(A * W)) <= 1.00001)

    # Make a global constraint matrix based on x-position

    if type(dthresh) is not str:
        C = np.eye(N, dtype=bool)
        dpos = abs(posx[:,None] - posx[None, :])
        C[dpos < dthresh] = True
    else:
        C = np.ones((N, N), dtype = bool)
    # Sample permutations for each worm
    perms = []
    Ps = np.zeros((M, N, N))
    for m in range(M):
        # perm[i] = index of neuron i in worm m's neurons
        perm = npr.permutation(N)
        perms.append(perm)
        Ps[m, np.arange(N), perm] = 1
        #Ps[m, np.arange(N),np.arange(N)] = 1
    # Make constraint matrices for each worm
    Cs = np.zeros((M, N, N), dtype=bool)

    for m, (Cm, Pm, permm) in enumerate(zip(Cs, Ps, perms)):
        # C is in canonical x canonical
        # make it canonical x worm[m] order
        Cm = C.dot(Pm)

        # Randomly choose a handful of given neurons
        given = npr.choice(N, replace=False, size=num_given)
        Cm[given, :] = 0
        Cm[:,permm[given]] = 0
        Cm[given,permm[given]] = 1
        Cs[m] = Cm
        assert np.sum(Pm * Cm) == N

    # Sample some data!
    Ys = np.zeros((M, T, N))
    for m in range(M):
        Ys[m,0,:] = np.ones(N)
        Wm = Ps[m].T.dot((W * A).dot(Ps[m]))
        for t in range(1, T):
            mu_mt = np.dot(Wm, Ys[m, t-1, :])
            Ys[m,t,:] = mu_mt + np.sqrt(etasq) * npr.randn(N)

    return Ys, A, W, Ps, Cs



def log_likelihood_single_worm(Y, A, W, P, etasq):
    N = A.shape[0]
    T = Y.shape[0]
    assert Y.shape == (T, N)
    assert A.shape == (N, N)
    assert W.shape == (N, N)
    assert P.shape == (N, N)

    Weff = np.dot(P.T, np.dot(W * A, P))
    Yerr = Y[1:] - np.dot(Y[:-1], Weff.T)
    ll = -0.5 * N * (T - 1) * np.log(2 * np.pi)
    ll += -0.5 * N * (T - 1) * np.log(etasq)
    ll += -0.5 * np.sum(Yerr ** 2 / etasq)
    return ll

def log_likelihood(Ys, A, W, Ps, etasq):
    # Compute log likelihood of observed data given W, Ps
    M = Ps.shape[0]
    ll = 0
    for m in range(M):
        ll += log_likelihood_single_worm(Ys[m], A, W, Ps[m], etasq)
    return ll


### MCMC
def run_naive_mcmc(Ys, A, Cs, etasq, sigmasq_W, W_true, Ps_true,
                   num_iters=500, num_mh_per_iter=1000,
                   W_init=None, Ps_init=None, do_update_W=True):
    # Iterate between solving for W | Ps and Ps | W
    M, T, N = Ys.shape
    assert A.shape == (N, N)

    # W = np.sqrt(sigmasq_W) * npr.randn(N, N)
    W = W_init if W_init is not None else np.sqrt(sigmasq_W) * npr.randn(N, N)

    # Initialize permutations and ensure they are valid
    Ps = Ps_init if Ps_init is not None else \
        np.array([perm_to_P(npr.permutation(N)) for _ in range(M)])
    for m, (P, C) in enumerate(zip(Ps, Cs)):
        P = round_to_perm(P - 1e8 * (1-C))
        assert np.sum(P[C]) == N
        Ps[m] = P


    sigmasq_W=10
    def _update_W(Ys, A, Ps, etasq):
        # Collect covariates
        Xs = []
        for Y, P in zip(Ys, Ps):
            Xs.append(np.dot(Y, P.T))
        X = np.vstack(Xs)

        W = np.zeros((N, N))
        for n in range(N):
            if np.sum(A[n]) == 0:
                continue

            xn = X[1:, n]
            Xpn = X[:-1][:, A[n]]
            Jn = np.dot(Xpn.T, Xpn) / etasq + sigmasq_W * np.eye(A[n].sum())
            Sign = np.linalg.inv(Jn)
            hn = np.dot(Xpn.T, xn) / etasq
            W[n, A[n]] = npr.multivariate_normal(np.dot(Sign, hn), Sign)
        return W

    # Identify the uncertain rows ahead of time
    def _naive_mh_step(Pm, Ym, A, W, Cm, curr_ll=None):
        # Randomly choose two neurons to swap
        unknowns = np.where(Cm.sum(axis=1) > 1)[0]
        n1, n2 = npr.choice(unknowns, 2, replace=False)
        v1 = np.where(Pm[n1])[0][0]
        v2 = np.where(Pm[n2])[0][0]
        if not Cm[n1, v2] or not Cm[n2, v1]:
            return Pm, curr_ll

        # Forward and Backward proposal probabilities are the same
        # so we just need to evaluate the log likelihoods
        curr_ll = curr_ll if curr_ll is not None else \
            log_likelihood_single_worm(Ym, A, W, Pm, etasq)

        P_prop = Pm.copy()
        P_prop[n1] = Pm[n2]
        P_prop[n2] = Pm[n1]
        prop_ll = log_likelihood_single_worm(Ym, A, W, P_prop, etasq)

        # Randomly accept or reject
        if np.log(npr.rand()) < prop_ll - curr_ll:
            return P_prop, prop_ll
        else:
            return Pm.copy(), curr_ll

    # Sample Pm | W with Metropolis Hastings
    def _update_Pm(Ym, A, W, Cm):
        Pm = Ps[m]
        curr_ll = None
        for _ in range(num_mh_per_iter):
            Pm, curr_ll = _naive_mh_step(Pm, Ym, A, W, Cm, curr_ll=curr_ll)
            # Pm, curr_ll = _smart_mh_step(Pm, Ym, A, W, Cm, curr_ll=curr_ll)

            # Check validity
            assert Pm[Cm].sum() == N
        return Pm

    lls = []
    mses = []
    num_corrects = []
    W_samples = []
    Ps_samples = []
    times = []

    def collect_stats(W, Ps):
        times.append(time.time())
        lls.append(log_likelihood(Ys, A, W, Ps, etasq) / (M * T * N))
        W_samples.append(W)
        Ps_samples.append(Ps)
        mses.append(np.mean((W * A - W_true * A) ** 2))

        # Round doubly stochastic matrix P to the nearest permutation matrix
        num_correct = np.zeros(M)
        for m, P in enumerate(Ps):
            row, col = linear_sum_assignment(-P + 1e8 * (1 - Cs[m]))
            num_correct[m] = n_correct(perm_to_P(col), Ps_true[m])
        num_corrects.append(num_correct)


    def callback(W, Ps, t):
        collect_stats(W, Ps)
        print("MCMC Iteration {}.  LL: {:.4f}  MSE(W): {:.4f}  Num Correct: {}"
              .format(t, lls[-1], mses[-1], num_corrects[-1]))

    # Run the MCMC algorithm
    callback(W, Ps, -1)
    for itr in range(num_iters):
        # Resample weights
        if do_update_W:
            W = _update_W(Ys, A, Ps, etasq)
        # Resample permutations
        for m in range(M):
            Ps[m] = _update_Pm(Ys[m], A, W, Cs[m])
        callback(W, Ps, itr)

    times = np.array(times)
    times -= times[0]



    return times, np.array(lls), np.array(mses), np.array(num_corrects)



### Variational inference

# Helpers to convert params into a random permutation-ish matrix
def perm_to_P(perm):
    K = len(perm)
    P = np.zeros((K, K))
    P[np.arange(K), perm] = 1
    return P

def round_to_perm(P):
    N = P.shape[0]
    assert P.shape == (N, N)
    row, col = linear_sum_assignment(-P)
    P = np.zeros((N, N))
    P[row, col] = 1.0
    return P

def n_correct(P1,P2):
    return P1.shape[0] - np.sum(np.abs(P1-P2))/2.0

def sinkhorn_logspace(logP, niters=10):
    for _ in range(niters):
        # Normalize columns and take the log again
        logP = logP - logsumexp(logP, axis=0, keepdims=True)
        # Normalize rows and take the log again
        logP = logP - logsumexp(logP, axis=1, keepdims=True)
    return logP

def make_map(C):
    assert C.dtype == bool and C.ndim == 2
    N1, N2 = C.shape
    valid_inds = np.where(np.ravel(C))[0]
    C_map = np.zeros((N1 * N2, C.sum()))
    C_map[valid_inds, np.arange(C.sum())] = 1

    def unpack_vec(v):
        return np.reshape(np.dot(C_map, v), (N1, N2))

    def pack_matrix(A):
        return A[C]

    return unpack_vec, pack_matrix

def initialize_params(A, Cs, map_W=None, map_Ps=None):
    N = A.shape[0]
    assert A.shape == (N, N)
    M = Cs.shape[0]
    assert Cs.shape == (M, N, N)

    unpack_W, pack_W = make_map(A)
    mu_W = np.zeros(A.sum()) if map_W is None else pack_W(map_W)
    log_sigmasq_W = -10 * np.ones(A.sum())

    log_mu_Ps = []
    log_sigmasq_Ps = []
    unpack_Ps = []

    for i,C in enumerate(Cs):
        unpack_P, pack_P = make_map(C)
        unpack_Ps.append(unpack_P)
        log_mu_Ps.append(
            np.zeros(C.sum()) if map_Ps is None else np.log(pack_P(map_Ps[i])+1e-8))
        log_sigmasq_Ps.append(-2 * np.ones(C.sum()))

    return mu_W, log_sigmasq_W, unpack_W, \
           log_mu_Ps, log_sigmasq_Ps, unpack_Ps


def initialize_params_gumbel(A, Cs, map_W=None, map_Ps=None):
    N = A.shape[0]
    assert A.shape == (N, N)
    M = Cs.shape[0]
    assert Cs.shape == (M, N, N)

    unpack_W, pack_W = make_map(A)
    mu_W = np.zeros(A.sum()) if map_W is None else pack_W(map_W)
    log_sigmasq_W = -10 * np.ones(A.sum())

    log_mu_Ps = []
    unpack_Ps = []
    for i,C in enumerate(Cs):
        unpack_P, pack_P = make_map(C)
        unpack_Ps.append(unpack_P)
        log_mu_Ps.append(
            np.zeros(C.sum()) if map_Ps is None else np.log(pack_P(map_Ps[i])+1e-8))

    return mu_W, log_sigmasq_W, unpack_W, \
           log_mu_Ps, unpack_Ps



def q_entropy(log_sigmasq_P, temp):
    return gaussian_entropy(0.5 * log_sigmasq_P) + log_sigmasq_P.size * np.log(temp)


def run_variational_inference_gumbel(Ys, A, W_true, Ps_true, Cs, etasq,stepsize=0.1,
                              init_with_true=True, num_iters=250, temp_prior=0.1, num_sinkhorn=20,
                              num_mcmc_samples=500, temp=1):

    def sample_q(params, unpack_W, unpack_Ps, Cs, num_sinkhorn, temp):

        # Sample W
        mu_W, log_sigmasq_W, log_mu_Ps  = params
        W_flat = mu_W + np.sqrt(np.exp(log_sigmasq_W)) * npr.randn(*mu_W.shape)

        W = unpack_W(W_flat)
        #W = W_true
        # Sample Ps: run sinkhorn to move mu close to Birkhoff
        Ps = []
        for log_mu_P , unpack_P, C in \
                zip(log_mu_Ps,  unpack_Ps, Cs):
            # Unpack the mean, run sinkhorn, the pack it again
            log_mu_P = unpack_P(log_mu_P)
            a = log_mu_P.shape
            log_mu_P = (log_mu_P + -np.log(-np.log(np.random.uniform(0, 1, (a[0], a[1])))))/temp

            log_mu_P = sinkhorn_logspace(log_mu_P - 1e8 * (1 - C), num_sinkhorn)
            log_mu_P = log_mu_P[C]

            ##Notice how we limit the variance
            P = np.exp(log_mu_P)
            P = unpack_P(P)

            Ps.append(P)

        Ps = np.array(Ps)
        return W, Ps



    def elbo(params, unpack_W, unpack_Ps, Ys, A, Cs, etasq, num_sinkhorn, num_mcmc_samples, temp_prior, temp):
        """
        Provides a stochastic estimate of the variational lower bound.
        sigma_Lim: limits for the variance of the re-parameterization of the permutation
        """

        def gumbel_distance(log_mu_Ps, temp_prior, temperature, Cs):
            arr = 0
            for n in range(len(log_mu_Ps)):
              log_mu_P = unpack_Ps[n](log_mu_Ps[n])
              C = Cs[n]
              log_mu_P=log_mu_P[C]
              log_mu_P=log_mu_P[:]
              arr+=np.sum(np.log(temp_prior) - 0.5772156649 * temp_prior / temperature -
              log_mu_P * temp_prior / temperature -
              np.exp(gammaln(
                 1 + temp_prior / temperature) - log_mu_P * temp_prior / temperature)
                          - (np.log(temperature) - 1 - 0.5772156649))
            return arr


        M, T, N = Ys.shape
        assert A.shape == (N, N)
        assert len(unpack_Ps) == M

        mu_W, log_sigmasq_W, log_mu_Ps = params

        L = 0

        for smpl in range(num_mcmc_samples):
            W, Ps = sample_q(params, unpack_W, unpack_Ps, Cs, num_sinkhorn, temp)

            # Compute the ELBO
            L += log_likelihood(Ys, A, W, Ps, etasq) / num_mcmc_samples

            L += gumbel_distance(log_mu_Ps, temp_prior, temp, Cs)
        # Add the entropy terms

        L += gaussian_entropy(log_sigmasq_W)
        fac = 1000
        ## This terms adds the KL divergence between the W prior and posterior with entries of W having a prior variance
        # sigma = 1/fac, for details see the appendix of the VAE paper.

        L += - 0.5 * log_sigmasq_W.size * (np.log(2 * np.pi)) -\
             0.5 * fac* np.sum(np.exp(log_sigmasq_W)) - 0.5 * fac * np.sum(
            np.power(mu_W, 2))
        # Normalize objective

        L /= (T * M * N)

        return L



    M, T, N = Ys.shape
    # Initialize variational parameters
    if init_with_true:
        mu_W, log_sigmasq_W, unpack_W, log_mu_Ps,  unpack_Ps = \
            initialize_params_gumbel(A, Cs,  map_W=W_true)
    else:
        mu_W, log_sigmasq_W, unpack_W, log_mu_Ps, unpack_Ps = \
            initialize_params_gumbel(A, Cs)

    # Make a function to convert an array of params into
    # a set of parameters mu_W, sigmasq_W, [mu_P1, sigmasq_P1, ... ]
    flat_params, unflatten = \
        flatten((mu_W, log_sigmasq_W, log_mu_Ps ))

    objective = \
        lambda flat_params, t: \
            -1 * elbo(unflatten(flat_params), unpack_W, unpack_Ps, Ys, A, Cs, etasq,
                      num_sinkhorn, num_mcmc_samples, temp_prior, temp)

    # Define a callback to monitor optimization progress
    elbos = []
    lls=[]
    mses = []

    num_corrects = []
    times = []



    W_samples = []
    Ps_samples = []



    def collect_stats(params, t):

        if t % 10 ==0:
            W_samples.append([])
            Ps_samples.append([])
            for i in range(100):
                W, Ps = sample_q(unflatten(params), unpack_W, unpack_Ps, Cs, num_sinkhorn,  temp)
                W_samples[-1].append(W)
                Ps_samples[-1].append(Ps)

        times.append(time.time())
        elbos.append(-1 * objective(params, 0))

        # Sample the variational posterior and compute num correct matches
        mu_W, log_sigmasq_W, log_mu_Ps = unflatten(params)

        W, Ps = sample_q(unflatten(params), unpack_W, unpack_Ps, Cs, 10, 1.0)




        list=[]
        for i in range(A.shape[0]):
            list.extend(np.where(Ps[0, i, :]+Ps_true[0, i, :] ==1)[0])

        mses.append(np.mean((W * A - W_true * A) ** 2))


        # Round doubly stochastic matrix P to the nearest permutation matrix
        num_correct = np.zeros(M)
        Ps2 = np.zeros((Ps.shape[0], A.shape[0], A.shape[0]))
        for m, P in enumerate(Ps):
            row, col = linear_sum_assignment(-P + 1e8 * (1 - Cs[m]))
            Ps2[m]= perm_to_P(col)
            num_correct[m] = n_correct(perm_to_P(col), Ps_true[m])
        num_corrects.append(num_correct)

        lls.append(log_likelihood(Ys, A, W, Ps2, etasq) / (M * T * N))

    def callback(params, t, g):
        collect_stats(params, t)
        print("Iteration {}.  ELBO: {:.4f} LL: {:.4f} MSE(W): {:.4f}, Num Correct: {}"
              .format(t, elbos[-1], lls[-1], mses[-1], num_corrects[-1]))

    # Run optimizer

    callback(flat_params, -1, None)
    variational_params = adam(grad(objective),
                              flat_params,
                              step_size=stepsize,
                              num_iters=num_iters,
                              callback=callback)

    times = np.array(times)
    times -= times[0]


    return times, np.array(elbos), np.array(lls), np.array(mses), \
           np.array(num_corrects), Ps_samples, W_samples, A, W_true




flags = tf.app.flags
gfile = tf.gfile
FLAGS = flags.FLAGS

flags.DEFINE_string('hparams','','hyperparameters')
flags.DEFINE_integer('num_iters', 5, 'Number of iterations')
flags.DEFINE_string('dir', 'test',
                    'Directory where to write event logs.')


DEFAULT_HPARAMS = tf.contrib.training.HParams(lr=0.1,
                                              type='MCMC',
                                              T=1000,
                                              M=1,
                                              num_given_neurons=10,
                                              dthresh=0.01,
                                              etasq=1.0,
                                              rho=0.4,
                                              temp_prior=1.0,
                                              temp=1.0,
                                              spectral_factor=1.1,
                                              num_iters=5,
                                              run=3)




def run_realistic_experiment(
        type, num_iters, M, T,
        num_given_neurons, dthresh, etasq, rho,
        spectral_factor, temp, temp_prior, lr, run, dir):
    npr.seed(run)

    experiment_name = "celegans_M{}_T{}_rho{}_giv{}_dthresh{}_etasq{}_sf{}". \
        format(M, T, rho, num_given_neurons, dthresh, etasq, spectral_factor)
    ## chose only a proportion rho of entire connectome
    props = np.ones((3, 4)) * rho
    As, names, xpos, = load_celegans_network(props)

    A = (np.sum(As, axis=0) > 0)
    A = (A+A.T)/2.0
    A = A.astype(bool)
    N = A.shape[0]

    RESULTS_DIR = os.path.join("results", dir)

    # Simulate a few "worm recordings"
    # sim = cached(RESULTS_DIR, experiment_name + "_data")(simulate_celegans)
    sim = simulate_celegans
    sigmasq_W = 1. / (1.5 * N * np.mean(A.sum(0)))

    Ys, A, W_true, Ps_true, Cs = \
        sim(A, xpos, M, T, num_given_neurons, dthresh, sigmasq_W, etasq, spectral_factor)
    print(np.max(abs(W_true)))
    print("Avg choices: {}".format(Cs.sum(1).mean()))
    print("E[W]: {:.4f}".format(W_true[A].mean()))
    print("Std[W]: {:.4f}".format(W_true[A].std()))
    print(experiment_name)
    if(type == 'VI_GUMBEL'):
    # Cached VI experiment function

      run_vi_gumbel = cached(
          RESULTS_DIR,
          experiment_name + "_vi_gumbel_temp=" +str(temp) +
          "temp_prior=" +str(temp_prior) +'lr=' +str(lr)+'run='+str(run))(run_variational_inference_gumbel)
      results_vi_gumbel = run_vi_gumbel(Ys, A, W_true, Ps_true, Cs,
                                        etasq, stepsize=lr, init_with_true=False, num_iters=num_iters,
                                        num_sinkhorn=10, num_mcmc_samples=1, temp_prior=temp_prior, temp=temp)
    if(type == 'MCMC'):

      #Cached MCMC experiment
      run_mcmc = cached(RESULTS_DIR, experiment_name + "_mcmc"+'run='+str(run))(run_naive_mcmc)
      results_mcmc = run_mcmc(
          Ys, A, Cs,etasq, sigmasq_W, W_true, Ps_true, num_iters=num_iters,
          W_init=None, do_update_W=True)



def main(_):
    hparams = DEFAULT_HPARAMS
    hparams.parse(FLAGS.hparams)
    type=hparams.type
    num_iters=hparams.num_iters
    M= hparams.M
    T= hparams.T
    num_given_neurons = hparams.num_given_neurons
    dthresh = hparams.dthresh
    etasq = hparams.etasq
    rho = hparams.rho
    spectral_factor = hparams.spectral_factor
    temp = hparams.temp
    temp_prior = hparams.temp_prior
    lr = hparams.lr
    run = hparams.run


    run_realistic_experiment(
        type, num_iters, M, T, num_given_neurons,
        dthresh, etasq, rho, spectral_factor,
        temp, temp_prior, lr, run, FLAGS.dir)
if __name__ == '__main__':
  tf.app.run(main)

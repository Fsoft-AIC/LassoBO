import math
import random
import botorch
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood, PredictiveLogLikelihood
from gpytorch.models import ExactGP
from benchmark import get_problem
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf

from abc import ABCMeta
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import standardize, normalize, unnormalize
from inner_optimizer import Turbo1_VS_Component
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from scipy.stats import norm

class GP(ExactGP, botorch.models.gpytorch.GPyTorchModel):
    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint, outputscale_constraint, ard_dims):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        base_kernel = RBFKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims)
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def _axis_score(f,train_x, train_y, hypers = None, prior = None, correct_hypers = False, num_restarts = 5):
    start_time = time.time()
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    bounds = torch.stack([torch.tensor(f.lb), torch.tensor(f.ub)])

    train_x = normalize(train_x,bounds)
    mu, sigma = train_y.median(), train_y.std()
    train_y = (train_y-mu)/(sigma+1e-6)

    noise_constraint = Interval(0, 5e-2)
    lengthscale_constraint = Interval(1e-3, 1e+6)
    outputscale_constraint = Interval(0.1, 10)

    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
    ard_dims = train_x.shape[1]
    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,)

    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    best = np.inf
    best_hypers = None
    #Find the best initialization of the hyperparameter
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if hypers is None:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.1
        hypers["covar_module.base_kernel.lengthscale"] = np.random.choice(a=np.array([1.,0.5,0.25]))
        hypers["likelihood.noise"] = 1e-2
        model.initialize(**hypers)
    else:
        model.load_state_dict(hypers)
    tol = 0
    for i in range(250):
        with gpytorch.settings.cholesky_jitter(1e-4):
            optimizer.zero_grad()
            output = model(train_x)
            lengthscale_params = model.covar_module.base_kernel.raw_lengthscale_constraint.transform(model.covar_module.base_kernel.raw_lengthscale)[0]
            l1_norm = sum(1./(p**2+1e-6) for k, p in enumerate(lengthscale_params) if k not in prior)
            loss = -mll(output, train_y) + 1e-4 * l1_norm
            loss.backward()
            optimizer.step()
            if loss.detach().item() < best:
                best = loss.detach().item()
                best_hypers = model.state_dict()
                tol = 0
            else:
                scheduler.step()
                tol = tol + 1
                if tol > 33:
                    break
            # if time.time() - start_time > 15:
            #     break
    model.eval()
    likelihood.eval()
    inv_ls = 1. / model.covar_module.base_kernel.lengthscale[0].cpu().detach().numpy()
    if correct_hypers:
        model.covar_module.outputscale = torch.clamp(2 * model.covar_module.outputscale, min = 1e-3, max=3.)
    return inv_ls , best_hypers, model
    
def shuffle_data(X):
    X_new = np.zeros_like(X)
    for i in range(X.shape[1]):
        X_new[:,i] = np.random.permutation(X[:,i])
    return X_new

def from_unit_cube(point, lb, ub):
    assert np.all(lb < ub) 
    assert lb.ndim == 1 
    assert ub.ndim == 1 
    assert point.ndim  == 2
    new_point = point * (ub - lb) + lb
    return new_point


def latin_hypercube(n, dims):
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n)) 
    centers = centers / float(2 * n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(n)]

    perturbation = np.random.uniform(-1.0, 1.0, (n, dims)) 
    perturbation = perturbation / float(2 * n)
    points += perturbation
    return points
def min_hypercube(n, dims):
    points = np.zeros((n, dims))
    k = int(np.floor(n ** (1./dims)) + 1)
    centers = (1.0 + 2.0 * np.arange(0.0, k)) 
    centers = centers / float(2 * k)
    for i in range(0, dims):
        points[:, i] = centers[np.random.choice(k,n)]
    perturbation = np.random.uniform(-1, 1., dims)/ (2 * k)
    points += perturbation
    return points

def upper_confidence_bound(gpr, X_sample, Y_sample, X, beta=0.25):
    X_sample = np.asarray(X_sample)
    Y_sample = np.asarray(Y_sample).reshape(-1, 1)
    mu, sigma = gpr.predict(X, return_std=True)
    Y_ucb = mu + beta * sigma
    return Y_ucb
def expected_improvement(gpr, X_sample, Y_sample, X, xi=0.0001, use_ei=True):

    X_sample = np.asarray(X_sample)
    Y_sample = np.asarray(Y_sample).reshape(-1, 1)
    mu, sigma = gpr.predict(X, return_std=True)

    if not use_ei:
        return mu
    else:
        #calculate EI
        mu_sample = gpr.predict(X_sample)
        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu_sample)
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            imp = imp.reshape((-1, 1))
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

def get_gpr_model(noise, length_scale, outputscale):
    rbf = ConstantKernel(outputscale) * RBF(length_scale=length_scale)
    gpr = GaussianProcessRegressor(kernel=rbf, alpha=noise**2)
    return gpr
def custom_optimize_acqf(dims, gpr, X_sample, Y_sample, active_dims, background):
    # maximize acquisition function
    inactive_dims = [k for k in range(dims) if k not in active_dims]
    beta = np.random.uniform(0.25, 1.6)
    X_cand  = []
    for i in range(background.shape[0]):
        X_new = latin_hypercube(10000, dims)
        X_new[:,inactive_dims] = background[i, inactive_dims]
        X_cand.append(X_new)
    X_cand = np.vstack(X_cand)
    start_time = time.time()
    acqf = upper_confidence_bound(gpr, X_sample, Y_sample, X_cand, beta)
    print("--- time for calculate ucb is %s seconds ---" % (time.time() - start_time))
    # X_acqf = upper_confidence_bound(gpr, X_sample, Y_sample, X, beta=0.1)
    acqf = acqf.reshape(-1)
    indices = np.argmax(acqf)
    proposed_X, proposed_X_acqf = X_cand[indices], acqf[indices]
    return proposed_X, proposed_X_acqf

class UiptStrategy(metaclass=ABCMeta):
    def __init__(self, dims, seed=42):
        self.dims = dims
        self.seed = seed
        
    def init_strategy(self, xs, ys):
        self.best_xs = []
        self.best_ys = []
        for x, y in zip(xs, ys):
            self.update(x, y)
    
    def get_full_variable(self, fixed_variables, lb, ub):
        pass
    
    def update(self, x, y):
        pass

class UiptRandomStrategy(UiptStrategy):
    def __init__(self, dims, seed=42):
        UiptStrategy.__init__(self, dims, seed)
        
    def get_full_variable(self, fixed_variables, lb, ub):
        new_x = np.zeros(self.dims)
        for dim in range(self.dims):
            if dim in fixed_variables.keys():
                new_x[dim] = fixed_variables[dim]
            else:
                new_x[dim] = np.random.uniform(lb[dim], ub[dim])
        return new_x
    def get_background(self, lb, ub, n=1):
        background = np.zeros((n, self.dims))
        for i in range(n):
            for dim in range(self.dims):
                background[i,dim] = np.random.uniform(lb[dim], ub[dim])
        return background

class UiptBestKStrategy(UiptStrategy):
    def __init__(self, dims, k=5, seed=42):
        UiptStrategy.__init__(self, dims, seed)
        self.k = k
        self.best_xs = []
        self.best_ys = []
    
    def get_full_variable(self, fixed_variables, lb, ub):
        best_xs = np.asarray(self.best_xs)
        best_ys = np.asarray(self.best_ys)
        new_x = np.zeros(self.dims)
        for dim in range(self.dims):
            if dim in fixed_variables.keys():
                new_x[dim] = fixed_variables[dim]
            else:
                new_x[dim] = np.random.choice(best_xs[:, dim])
        return new_x

    def get_background(self, lb, ub, n=2):
        best_xs = np.asarray(self.best_xs)
        best_ys = np.asarray(self.best_ys)
        background = np.zeros((n, self.dims))
        for i in range(n):
            for dim in range(self.dims):
                background[i,dim] = np.clip(np.random.choice(best_xs[:, dim]) + 0.05 *np.random.choice(2) * np.random.randn(),a_min = lb[dim], a_max=ub[dim])
        return background
    
    def update(self, x, y):
        if len(self.best_xs) < self.k:
            self.best_xs.append(x)
            self.best_ys.append(y)
            if len(self.best_xs) == self.k:
                self.best_xs = np.vstack(self.best_xs)
                self.best_ys = np.array(self.best_ys)
        else:
            min_y = np.min(self.best_ys)
            if y > min_y:
                idx = np.random.choice(np.argwhere(self.best_ys == min_y).reshape(-1))
                self.best_xs[idx] = x
                self.best_ys[idx] = y
        assert len(self.best_xs) <= self.k
def local_search_sampling(f,train_x, train_y,active_dims, ddirs):
    s, n = train_x.shape
    inactive_dims = [i for i in range(n) if i not in active_dims]
    k = train_y.argmax()
    new_x = train_x[k].copy()
    new_dir = 1e-2*(f.ub-f.lb)*np.random.randn()
    for i in range(n):
        if i in inactive_dims:
            for ddir in ddirs:
                new_dir[i] = new_dir[i] + ddir[i]
                new_x[i] = np.clip(new_x[i] + new_dir[i],a_min=f.lb[i],a_max=f.ub[i])
        else:
            new_dir[i] = 0
    new_dir = new_dir/(len(ddirs)+1)
    new_y = f(new_x)
    return new_x, new_y, new_dir

def BO_torch(f, model, train_x, train_y,active_dims, rd_background = None, correct_hypers = False):
    n_active_dims = len(active_dims)
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    bounds = torch.stack([torch.tensor(f.lb), torch.tensor(f.ub)])

    train_x = normalize(train_x,bounds)
    mu, sigma = train_y.median(), train_y.std()
    train_y = (train_y-mu)/(sigma+1e-6)
    train_y = train_y.reshape(-1,1)
    if rd_background is None :
        rd_background = latin_hypercube(2, f.dims)

    uipt_solver = UiptBestKStrategy(dims = f.dims, k = 3)
    uipt_solver.init_strategy(train_x.numpy(),train_y.numpy())
    best_background = uipt_solver.get_background(np.zeros(f.dims),np.ones(f.dims))
    background = np.vstack([rd_background, best_background])


    likelihood = deepcopy(model.likelihood)
    mean_module = deepcopy(model.mean_module)
    covar_module = deepcopy(model.covar_module)

    if correct_hypers:
        covar_module.outputscale = torch.clamp(1.1 * covar_module.outputscale, min = 1e-3, max=3.)
        covar_module.base_kernel.lengthscale = torch.clamp(covar_module.base_kernel.lengthscale / 1.05, min= 1e-2, max=1e+3)

    gp = SingleTaskGP(train_x, train_y, likelihood = likelihood, covar_module = covar_module, mean_module = mean_module)
    beta = 0.75 + 0.75 * np.random.rand()
    # if np.random.rand() < 0.09:
    #     beta = 3
    UCB = UpperConfidenceBound(gp, beta = beta)
    # EI  = ExpectedImprovement(gp, best_f=train_y.max()+ 1e-3)

    std_bounds = torch.stack([torch.zeros(f.dims), torch.ones(f.dims)])

    max_acq_value = -np.inf
    new_x = None
    start_time = time.time()
    for i in range(background.shape[0]):
        fixed_features = {j: background[i,j] for j in range(f.dims) if j not in active_dims}
        candidate, acq_value = optimize_acqf(UCB, bounds=std_bounds, q=1, fixed_features= fixed_features, num_restarts=8, raw_samples=1000,)
        if max_acq_value < acq_value.item():
            new_x = unnormalize(candidate, bounds)[0]
            max_acq_value = acq_value.item()
    new_x = new_x.detach().numpy()
    print("-----beta is %s--------" % beta)
    print("--- time for optimizing acqf is %s seconds ---" % (time.time() - start_time))
    new_y = f(new_x)
    return new_x, new_y

# def BO_numpy(f, model, train_x, train_y, active_dims, rd_background = None, correct_hypers = False):
#     train_x = torch.from_numpy(train_x)
#     train_y = torch.from_numpy(train_y)
#     bounds = torch.stack([torch.tensor(f.lb), torch.tensor(f.ub)])

#     train_x = normalize(train_x,bounds).numpy()
#     train_y = standardize(train_y).numpy()
#     if rd_background is None :
#         rd_background = latin_hypercube(30, f.dims)

#     uipt_solver = UiptBestKStrategy(dims = f.dims, k = 2)
#     uipt_solver.init_strategy(train_x,train_y)
#     best_background = uipt_solver.get_background(np.zeros(f.dims),np.ones(f.dims))
#     background = np.vstack([rd_background, best_background])


#     likelihood = deepcopy(model.likelihood)
#     mean_module = deepcopy(model.mean_module)
#     covar_module = deepcopy(model.covar_module)

#     outputscale = np.clip(1.1 * covar_module.outputscale.detach().numpy(), a_min = 1e-3, a_max=2.)
#     length_scale = np.clip(covar_module.base_kernel.lengthscale.detach().numpy() / 1.05, a_min= 5e-2, a_max=1e+4)
#     noise = float(likelihood.noise)
    
#     gpr = get_gpr_model(noise, length_scale.reshape(f.dims).min(), outputscale)
#     start_time = time.monotonic()
#     gpr.fit(train_x, train_y)
#     print("--- time for fit gp is %s seconds ---" % (time.monotonic() - start_time))
#     proposed_X, _ = custom_optimize_acqf(f.dims, gpr, train_x, train_y, active_dims, background)
#     proposed_X = proposed_X * (f.ub - f.lb) + f.lb
#     proposed_y = f(proposed_X)
#     return proposed_X, proposed_y


def find_num_ads(scores, alpha = 0.9, window_size= 1, dim = 100):
    d = 0
    window_size = np.min([len(scores),window_size])
    score = np.sum(scores[-window_size:],axis =0)
    full_sum = np.sum(score)
    score = np.sort(score)[::-1]
    selected_sum = 0
    for s in score:
        selected_sum = selected_sum + s
        d = d + 1
        if selected_sum >= alpha * full_sum:
            break
    return np.clip(d, a_max= int(dim* 0.6), a_min=1)
def run_LassoBO(f, n_init, max_evals):
    dim = len(f.lb)
    X_init = latin_hypercube(n_init, dim)
    X_init = from_unit_cube(X_init, f.lb, f.ub)
    fX_init = np.array([f(x) for x in X_init])
    id_best = fX_init.argmax()
    best_f = fX_init[id_best]
    hypers = None
    is_train = True
    tol = 0
    n_iter = max_evals
    batch_size = 1
    ipt_solver = 'bo'
    M = 5
    max_tol = M * 2 + 1
    rd_background = latin_hypercube(M, f.dims)
    rec_scores = []
    prior = np.array([])
    for iter in range(n_iter):
        if is_train:
            score, hypers, model = _axis_score(f,X_init, fX_init, None, [])
            rec_scores = []
            rd_background = latin_hypercube(M, f.dims)
        else: 
            score, hypers, model = _axis_score(f,X_init, fX_init, hypers, [])
            rd_background = shuffle_data(rd_background)
        is_train = False
        ucb = score.copy()
        rec_scores.append(score)
        d = find_num_ads(rec_scores, 0.9, dim=dim)
        ads = np.argsort(-ucb)[:d]
        for ned in ads:
            if ned not in prior:
                break 
        prior = np.unique(np.append(np.intersect1d(prior, ads),ned)) 
        ads.sort()
        if ipt_solver=='bo':
            for i in range(batch_size):
                new_x, new_y = BO_torch(f, model, X_init, fX_init, ads.tolist(), rd_background= rd_background)
                if new_y > best_f:
                    best_f = new_y
                    id_best = len(fX_init)
                    tol = 0
                else:
                    tol = tol + 1
                    if tol > max_tol:
                        tol = 0
                        is_train = True
                X_init = np.append(X_init,[new_x],axis = 0)
                fX_init = np.append(fX_init, new_y)
                print(new_y)
        else:
            assert 0




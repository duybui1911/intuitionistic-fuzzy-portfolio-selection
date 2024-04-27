import time
import argparse
from tqdm import tqdm
import numpy as np
from autograd import grad
import scipy.optimize as optimize
from portfolio import Portfolio
import utils
import warnings
warnings.filterwarnings("ignore")

def rosen(x,y):
    """The Rosenbrock function"""
    return np.sqrt(np.sum((x-y)**2))

def constraint(x):
    x = np.array(x)
    A = np.ones((1, len(x)))
    b = np.array([[1.0]])
    return (A @ (x.T) - b.T).tolist()[0][0]

def find_projection(y, n, Ex=None, alpha=None, Varx=None, beta=None):
    under = np.zeros(n).tolist()
    upper = np.ones(n).tolist()
    bounds = optimize.Bounds(under, upper)
    x = np.random.rand(1, n).tolist()[0]

    if Ex is not None and Varx is not None:
        Ex_grad = grad(Ex)
        Varx_grad = grad(Varx)
    
    cons = (
        {
            "type": "eq",
            "fun": lambda x: np.array([constraint(x)]),
            "jac": lambda x: np.ones((1, len(x))),
        },
        # {
        #     "type": "ineq",
        #     "fun": lambda x: np.array([-(-Ex(x) - alpha)]),
        #     "jac": lambda x: np.array([Ex_grad(x)]),
        # },
        # {
        #     "type": "ineq",
        #     "fun": lambda x: np.array([-Varx(x) + beta]),
        #     "jac": lambda x: np.array([-Varx_grad(x)]),
        # },
    )
    res = optimize.minimize(
            rosen, x, jac="2-point", args=(y), hess=optimize.BFGS(),
            constraints=cons, method="trust-constr", options={"disp": False}, bounds=bounds,
        )
    return res.x

def solve(f, f_dx, x: float, lda:float, sigma:float, K:float, max_iters: int, n: int, Ex, alpha, Varx, beta):
    res = [x]
    val = [np.array([f(x)])]
    list_lda = []
    list_dfx = []
    x_pre = x
    for t in tqdm(range(max_iters), desc='Solving problem: '):
        y = x - lda*f_dx(x)
        list_dfx.append(f_dx(x))
        x_pre = x.copy()
        x = find_projection(y, n, Ex, alpha, Varx, beta)
        if f(x) - f(x_pre) + sigma*(np.dot(f_dx(x_pre).T, x_pre - x)) <= 0:
            lda = lda
        else:
            if lda > 0.05:
                lda = K*lda
        
        res.append(x)
        val.append(np.array([f(x)]))
        list_lda.append(np.array([lda]))
    return res, val, list_lda, list_dfx


def parse_args():
    parser  = argparse.ArgumentParser(description='Portfolio Selection ICIT-2023')
    parser.add_argument('--ex1', action='store_true', default=False)
    parser.add_argument('--ex2', action='store_true', default=False)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    alpha = 0.25
    beta = 2.5
    max_iters1 = 200

    if args.ex1:
        print('Solving Ex1: ')
        L = np.array([0.0282, 0.0462, 0.0188, 0.0317, 0.01536, 0.0097, 0.01919])
        Q = np.array([[0.0119, 0.0079, 0.0017, 0.0019, 0.0022, -0.0008, 0.0032],
              [0.0079, 0.0157, 0.0016, 0.0013, 0.0005, -0.0026, 0.0035],
              [0.0017, 0.0016, 0.0056, -0.0002, 0.0030, 0.0017, -0.0003],
              [0.0019, 0.0013, -0.0002, 0.0093, -0.0007, 0.0010, 0.0024],
              [0.0022, 0.0005, 0.0030, -0.0007, 0.0110, 0.0010, 0.0011],
              [-0.0008, -0.0026, 0.0017, 0.0010, 0.0010, 0.0067,0.0014],
              [0.0032, 0.0035, -0.0003, 0.0024, 0.0011, 0.0014, 0.0130]])
        pf = Portfolio(L=L, Q = Q, p_rf=0.005)
    elif args.ex2:
        print('Solving Ex2: ')
        pf = Portfolio(data_path='data_500.txt')
    else:
        assert False, 'Please select a problem [ex1, ex2].'
    
    print('L vector: ', pf.L)
    print('Q matrix: ', pf.Q)
    # init objective
    dict_problem = {
        'MV': pf.MV,
        'MVS': pf.MVS,
        'fuzzy_MV': pf.fuzz_MV,
        'fuzz_MVS': pf.fuzz_MVS
    }

    for pb_key in dict_problem.keys():
        print('Problem: ', pb_key)
        n = len(pf.L)

        x0 = find_projection(np.random.rand(1, n), n, None, None, None, None) # init point
        print("\tInit x0: ", x0)
        
        Lambda1 = 1. # init lambda in (0, +vc)
        sigma = 0.1 # init sigma in (0,1)
        K = 0.9 # init scale_down ratio of step_size (0,1)

        t1 = time.time()
        fx = dict_problem[pb_key]
        fx_dx = grad(fx)
        res1, val1, lda1, dfx = solve(
            fx, 
            fx_dx, 
            x0, 
            Lambda1, 
            sigma, 
            K, 
            max_iters1, 
            n,
            # pf.Ex, 
            # alpha, 
            # pf.Varx, 
            # beta
            None, None, None, None
        )
        t2 = time.time()
        tmp1 = np.array(res1)[:,:]
        # old_sol = tmp1[:-1]
        print('\tSolution: ', tmp1[-1])
        print('\tE(x): ', pf.Ex(tmp1[-1]))
        print('\tVar(x): ', pf.Varx(tmp1[-1]))
        print('\tSharp(x): ', pf.Sr(tmp1[-1]))
        print("\tTime to find solutions: ",t2-t1)
        lda1 = np.array(lda1)
        dfx = np.array(dfx)
        val1 = np.array(val1)
        
        # Plot trajectory
        utils.plot_x(tmp1, f'outputs/quydaonghiem_{pb_key}.png', key='x')
        utils.plot_x(lda1, f'outputs/lda_{pb_key}.png', key='lambda')
        utils.plot_x(dfx, f'outputs/grad_{pb_key}.png', key='dfx')
        utils.plot_x(val1, f'outputs/value_{pb_key}.png', key='f')



if __name__ == '__main__':
    main()
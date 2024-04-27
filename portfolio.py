import numpy as np
import utils
from typing import Union
from autograd import grad
import scipy.optimize as optimize

class Portfolio(object):
    def __init__(self,  
        L: Union[np.ndarray, None]=None, 
        Q: Union[np.ndarray, None]=None, 
        data_path: Union[str, None]=None,
        p_rf: float=0.022,
    ):
        '''
            This class handles calculations and attributes related to a portfolio. 
            It can be initialized with the following parameters:
                - L: A vector representing the expected returns of assets.
                - Q: A covariance matrix describing the relationships between asset returns.
                - data_path: The path to a text file containing stock data arranged 
                in columns similar to the file data_500.txt, if you don't provide L and Q.
                - p_rf: the rate of a zero-risk portfolio's return.
        '''
        if L is None or Q is None:
            self.L, self.Q =  self.load_and_pre_process_data(data_path)
        else:
            self.L = L
            self.Q = Q
        self.p_rf = p_rf
        self.Ex_star_min, self.Ex_star_max, self.Vx_min,\
        self.Vx_max, self.Sr_star_min, self.Sr_star_max = self.find_bounds_E_V_S()
     
    def MV(self, x):
        return max(self.Ex_star(x), self.Varx(x))

    def MVS(self, x):
        return max(self.Ex_star(x), self.Varx(x), self.Sr_star(x))

    def fuzz_MV(self, x):
        # mv_values = [self.Ex_star(x), self.Varx(x)]
        fuzzy_mv_values = (
            1. - utils.linear_membership(self.Ex_star(x), self.Ex_star_min, self.Ex_star_max),
            1. - utils.linear_membership(self.Varx(x), self.Vx_min, self.Vx_max),
            utils.linear_nonmembership(self.Ex_star(x), self.Ex_star_min, self.Ex_star_max),
            utils.linear_nonmembership(self.Varx(x), self.Vx_min, self.Vx_max),
        )
        return max(fuzzy_mv_values)

    def fuzz_MVS(self, x):
        # # mvs_values = [self.Ex_star(x), self.Varx(x), self.Sr_star(x)]
        fuzzy_mvs_values = (
            1. - utils.linear_membership(self.Ex_star(x), self.Ex_star_min, self.Ex_star_max),
            1. - utils.linear_membership(self.Varx(x), self.Vx_min, self.Vx_max),
            1. - utils.linear_membership(self.Sr_star(x), self.Sr_star_min, self.Sr_star_max),
            utils.linear_nonmembership(self.Ex_star(x), self.Ex_star_min, self.Ex_star_max),
            utils.linear_nonmembership(self.Varx(x), self.Vx_min, self.Vx_max),
            utils.linear_nonmembership(self.Sr_star(x), self.Sr_star_min, self.Sr_star_max),
        )
        return max(fuzzy_mvs_values)

    def Ex(self, x):
        x = np.array(x)
        return np.dot(self.L, np.transpose(x))
    
    def Ex_star(self, x):
        return - self.Ex(x)

    def Varx(self, x):
        x = np.array(x)
        return np.dot(np.transpose(x), np.dot(self.Q, x))

    def Varx_star(self, x):
        return - self.Varx(x)

    def Sr(self, x):
        x = np.array(x)
        try:
            return (self.Ex(x) - self.p_rf)/np.sqrt(self.Varx(x))
        except:
            return (self.Ex(x) - self.p_rf)/np.sqrt(self.Varx(x)._value)
    
    def Sr_star(self, x):
        return - self.Sr(x)

    def find_bounds_E_V_S(self,):
        under = np.zeros_like(self.L).tolist()
        upper = np.ones_like(self.L).tolist()
        bounds = optimize.Bounds(under, upper)
        E_dx = grad(self.Ex)
        V_dx = grad(self.Varx)
        S_dx = grad(self.Sr)

        E_star_dx = grad(self.Ex_star)
        V_star_dx = grad(self.Varx_star)
        S_star_dx = grad(self.Sr_star)
        g3x_dx = grad(self.g3x)

        x0 = np.ones_like(self.L)
        cons = (
            {
                "type": "eq",
                "fun": lambda x: np.array([-self.g3x(x)]),
                "jac": lambda x: np.array([-g3x_dx(x)]),
            },
        )

        sol_min_Ex_star = optimize.minimize(
            self.Ex_star, x0, jac="2-point", hess=optimize.BFGS(),
            constraints=cons, method="trust-constr", options={"disp": False}, bounds=bounds,
        )
        sol_max_Ex_star = optimize.minimize(
            self.Ex, x0, jac="2-point", hess=optimize.BFGS(),
            constraints=cons, method="trust-constr", options={"disp": False}, bounds=bounds,
        )

        sol_min_Vx = optimize.minimize(
            self.Varx, x0, jac="2-point", hess=optimize.BFGS(),
            constraints=cons, method="trust-constr", options={"disp": False}, bounds=bounds,
        )
        sol_max_Vx = optimize.minimize(
            self.Varx_star, x0, jac="2-point", hess=optimize.BFGS(),
            constraints=cons, method="trust-constr", options={"disp": False}, bounds=bounds,
        )
        sol_min_Sr_star = optimize.minimize(
            self.Sr_star, x0, jac="2-point", hess=optimize.BFGS(),
            constraints=cons, method="trust-constr", options={"disp": False}, bounds=bounds,
        )
        sol_max_Sr_star = optimize.minimize(
            self.Sr, x0, jac="2-point", hess=optimize.BFGS(),
            constraints=cons, method="trust-constr", options={"disp": False}, bounds=bounds,
        )

        Ex_star_min = self.Ex_star(sol_min_Ex_star.x)
        Ex_star_max = self.Ex_star(sol_max_Ex_star.x)

        Vx_min = self.Varx(sol_min_Vx.x)
        Vx_max = self.Varx(sol_max_Vx.x)

        Sr_star_min = self.Sr_star(sol_min_Sr_star.x)
        Sr_star_max = self.Sr_star(sol_max_Sr_star.x)

        list_return = [
            Ex_star_min,
            Ex_star_max,
            Vx_min,
            Vx_max,
            Sr_star_min,
            Sr_star_max
        ]
        return list_return

    def mu0(self, x):
        if self.Ex_star(x) < self.Ex_star_min:
            return 1.
        elif self.Ex_star(x) > self.Ex_star_max:
            return 0.
        else:
            return (self.Ex_star_max - self.Ex_star(x))/(self.Ex_star_max - self.Ex_star_min)

    def nu0(self, x):
        if self.Ex_star(x) < self.Ex_star_min:
            return 0.
        elif self.Ex(x) > self.Ex_star_max:
            return 1.
        else:
            return (self.Ex(x) - self.Ex_star_min)/(self.Ex_star_max - self.Ex_star_min)
    
    def mu1(self, x):
        if self.Varx(x) < self.Vx_min:
            return 1.
        elif self.Varx(x) > self.Vx_max:
            return 0.
        else:
            return (self.Vx_max - self.Varx(x))/(self.Vx_max - self.Vx_min)

    def nu1(self, x):
        if self.Varx(x) < self.Vx_min:
            return 0.
        elif self.Varx(x) > self.Vx_max:
            return 1.
        else:
            return (self.Varx(x) - self.Vx_min)/(self.Vx_max - self.Vx_min)
    
    def mu2(self, x):
        if self.Sr_star(x) < self.Sr_star_min:
            return 1.
        elif self.Sr_star(x) > self.Sr_star_max:
            return 0.
        else:
            return (self.Sr_star_max - self.Sr_star(x))/(self.Sr_star_max - self.Sr_star_min)

    def nu2(self, x):
        if self.Sr_star(x) < self.Sr_star_min:
            return 0.
        elif self.Sr_star(x) > self.Sr_star_max:
            return 1.
        else:
            return (self.Sr_star(x) - self.Sr_star_min)/(self.Sr_star_max - self.Sr_star_min)
    
    
    @staticmethod
    def constraint(x):
        x = np.array(x)
        A = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
        b = np.array([[1.0]])
        return (A @ (x.T) - b.T).tolist()[0][0]
    @staticmethod
    def g3x(x):
        '''
            Constraint: x1 + x2 + ... + xn = 1
        '''
        x = np.array(x)
        return np.sum(x) - 1
    
    @staticmethod
    def load_and_pre_process_data(name_file) -> tuple:
        stock_prices = np.loadtxt(name_file)
        n_data, num_stock = stock_prices.shape[:2]
        stock_prices = np.transpose(stock_prices)
        temp = np.zeros((num_stock, n_data - 1))
        for i in range(num_stock):
            for j in range(n_data - 1):
                temp[i, j] = stock_prices[i, j + 1] / stock_prices[i, j] - 1

        temp = 100 * temp
        L = np.sum(temp, 1) / (temp.shape[1]-1)
        Q = np.cov(temp)
        return (np.array(L), np.array(Q))
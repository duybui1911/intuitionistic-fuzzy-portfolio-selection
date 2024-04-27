import numpy as np
from autograd import grad
import matplotlib.pyplot as plt

def linear_membership(curr_value, min_value, max_value):
    if curr_value < min_value:
      return 1.
    elif curr_value > max_value:
      return 0.
    else:
      return (max_value - curr_value)/(max_value - min_value)

def linear_nonmembership(curr_value, min_value, max_value, delta=0.0):
    return 1 - linear_membership(curr_value, min_value, max_value) - delta

def plot_x(sol_all, filename, show=False, key='x'):
    t = [i for i in range(len(sol_all))]
    plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size': 16})
    n_var = len(sol_all[0])
    
    if n_var > 1:
      for j in range(n_var):
        plt.plot(t, sol_all[:, j], label=r'$%s_{%d}(t)$' % (key, j+1),linewidth=1)
      plt.legend([r'$%s_{%d}(t)$' % (key, j+1) for j in range(n_var)])
    else:
      plt.plot(t, sol_all, label= r'$%s(t)$' % key ,linewidth=1)
      plt.legend([r'$%s(t)$' % key])
    plt.xlabel('iteration')
    plt.ylabel('%s(t)' % key)
    plt.savefig(filename)
    if show:
        plt.show()
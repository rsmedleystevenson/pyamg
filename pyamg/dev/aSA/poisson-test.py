import sys

import matplotlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation

import numpy
from numpy import ones, ravel, arange, mod, loadtxt, array, dot, abs, \
                  zeros_like, sqrt, meshgrid, linspace, pi, sin

import scipy
from scipy.linalg import pinv, pinv2
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, isspmatrix_bsr
from scipy import rand, zeros, hstack, cos, sin, pi, vstack, mat, sparse, log10

import pyamg
from pyamg import relaxation, gallery, util
from pyamg.relaxation.relaxation import gauss_seidel, gauss_seidel_indexed
from pyamg.util.linalg import norm
from pyamg.util.utils import scale_rows, scale_columns
from pyamg.gallery import stencil_grid
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.util.utils import symmetric_rescaling

import aSA
from aSA import A_norm, my_rand, asa_solver

from copy import deepcopy
from pyamg.multilevel import multilevel_solver

# TODO:
# Does w cycles get worse as we do more iterations
# Ideally solve coarse grid exactly and compare to v
# construct full hierarchy but run it as two level
#
# Force local ritz to 1 target vector
#
# start with sin hump target.. see what happens. Should remain the same

###############################################################################
# Parameters
###############################################################################

# Problem parameters
n = 300
epsilon = 1.
theta = 0.
# randomly weight the columns of A
randomize = False
# scale A by its diagnol
symmetric_scale = True

# Prolongation smoother
smooth = ('richardson', {'omega': 1.})
# smooth = ('jacobi', {'omega':1})

# Relaxation smoother
prepostsmoother = ('richardson', {'omega': 3./2.})
# prepostsmoother = ('jacobi', {'iterations':2, 'omega':2./3.})

# Coarse level parameters
coarse_solver = 'splu'
max_coarse = n*n*0.5
coarse_size = 30

# Aggregation
strength = 'symmetric'
aggregate = 'standard'

# Solver test parameters
maxiter = 50
tol = 1e-16
max_levels = 30
max_level_iterations = 5

# Adaptation parameters
max_targets = 10
min_targets = 0
num_initial_targets = 1
targets_iters = 15
conv_tol=0.2
weak_tol= 15 # increasing this makes it more likely to remove global targets
local_weak_tol= 15 # increasing this makes it more likely to remove local targets
initial_targets=None

plot_targets = True # should we plot targets?

###############################################################################
# Define matrix function that returns a dictionary which includes among others:
# A, X, Y, b, and x0. This is some versions of 2D Poisson.
###############################################################################

def build_poisson(n, epsilon, theta, randomize):
    data = {}
    h = 1./float(n+1)

    print "Assembling diffusion using Q1 on a regular mesh with epsilon = " + \
          str(epsilon) + " and theta = " + str(theta) + " ..."
    stencil = diffusion_stencil_2d(type='FE', epsilon=epsilon, theta=theta)
    A = stencil_grid(stencil, (n, n), format='csr')
    X,Y = meshgrid(linspace(h, 1.0 - h, n), linspace(h, 1.0 - h, n))
    data['X'] = X
    data['Y'] = Y

    if randomize:
        print "Random diagonal scaling..."
        D = my_rand(A.shape[0], 1)
        D[D < 0.] -= 1e-3
        D[D >= 0.] += 1e-3
        data['D'] = D
        D_inv = 1./D
        data['D_inv'] = D_inv
        A = scale_rows(A, D)
        A = scale_columns(A, D)

    if symmetric_scale:
        symmetric_rescaling(A, copy=False)

    print "Ratio of largest to smallest (in magnitude) diagonal element in A: %1.3e"% \
          (abs(A.diagonal()).max() / abs(A.diagonal()).min())
    data['A'] = A
    print 'Generate initial guess (which is random)...'
    data['x0'] = my_rand(data['A'].shape[0], 1)
    print 'Generate rhs (which is zero)...'
    data['b'] = numpy.zeros((data['A'].shape[0], 1))

    return data

###############################################################################
# Construct the problem and adjust some parameters
###############################################################################

data = build_poisson(n, epsilon, theta, randomize)
A = data['A'];
b = data['b'];
x0 = data['x0'];
X = data['X']
Y = data['Y']
N = A.shape[0]
print 'Number of unknowns: ' + str(N)

## Constant initial target
#initial_targets = ones((N,1))

## Single sine hump initial target
#initial_targets = (sin(X*pi)*sin(Y*pi)).reshape(-1,1)

###############################################################################
# Print out parameters
###############################################################################

print "\n################## Parameters ##################"
print 'Problem parameters:'
print '    Grid points in one direction (n): ' + str(n)
print '    Anisotropy parameter (epsilon): ' + str(epsilon)
print '    Anisotropy rotation angle (theta): ' + str(theta)
print '    Random scaling (randomize): ' + str(randomize)
print '    Diagonal scaling: ' + str(symmetric_scale)
print ''
print 'Solver parameters:'
print '    Prolongation smoother (smooth): ' + str(smooth)
print '    Relaxation smoother (prepostsmoother): ' + str(prepostsmoother)
print '    Coarse solver (coarse_solver): ' + str(coarse_solver)
print '    Maximal # coarse dofs (max_coarse): ' + str(max_coarse)
print '    # coarse dofs for coarse solve (coarse_size): ' + str(coarse_size)
print '    Strength of connections (strength): ' + str(strength)
print '    Aggregation (aggregate): ' + str(aggregate)
print ''
print 'Solver test parameters:'
print '    Maximal number of iterations (maxiter): ' + str(maxiter)
print '    Relative residual tolerance (tol): ' + str(tol)
print '    Maximum number of levels: ' + str(max_levels)
print '    Maximum number of iterations on a level: ' + str(max_level_iterations)
print ''
print 'Adaptation parameters:'
print '    Maximal number of targets (max_targets): ' + str(max_targets)
print '    Maximal number of targets (min_targets): ' + str(min_targets)
print '    # iterations for computing a target (targets_iters): ' + str(targets_iters)
print '    Target convergence rate (conv_tol): ' + str(conv_tol)
print '    Starting constant in WAP (weak_tol): ' + str(weak_tol)
print '    Starting constant in local WAP (local_weak_tol): ' + str(local_weak_tol)
print '    Initial number of targets to generate: ' + str(num_initial_targets)
print '    Initial set of targets (initial_targets): ' + str(initial_targets)
print "################################################\n"

###############################################################################
# Build the solver
###############################################################################

print "\nBuilding the solver...\n"
[solver, work] = asa_solver(A, initial_targets=initial_targets,
                              max_targets=max_targets,
                              min_targets=min_targets,
                              num_initial_targets=num_initial_targets,
                              targets_iters=targets_iters, conv_tol=conv_tol,
                              weak_tol=weak_tol, local_weak_tol=local_weak_tol,
                              max_coarse=max_coarse, max_levels=max_levels,
                              max_level_iterations=max_level_iterations,
                              prepostsmoother=prepostsmoother,
                              smooth=smooth, strength=strength, aggregate=aggregate,
                              coarse_solver=coarse_solver, coarse_size=coarse_size,
                              verbose=True)
print "\nSolver is constructed. Estimated construction work: %1.3f"%work
print "Final solver:"
print solver

print 'Targets in use: ' + str(solver.levels[0].B.shape[1])

###############################################################################
# Test the solver
###############################################################################

print "\nInitial error energy: ||x0|| = %1.5e\n"%(A_norm(x0, A))

prev_err = 0.
curr_it = 0
def callback(A, xk):
    global prev_err
    global curr_it
    curr_it += 1
    curr_err = A_norm(xk, A)
    output = "Iteration #%3d: error energy ||xk|| = %1.5e"%(curr_it, curr_err)
    if curr_it > 1:
        output += ", convergence factor = %1.5f"%(curr_err/prev_err)
    print output
    prev_err = curr_err

xn = solver.solve(b, x0=x0, tol=tol, maxiter=maxiter, cycle='V', callback=lambda x: callback(A,x))
# A2 = solver.levels[1].A
# solver.levels = [deepcopy(solver.levels[0]), multilevel_solver.level()]
# solver.levels[1].A = A2
# xn = solver.solve(b, x0=x0, tol=tol, maxiter=maxiter, cycle='V', callback=lambda x: callback(A,x))

print "\nFinal error energy: ||xn|| = %1.5e\n"%(A_norm(xn, A))

# A2 = solver.levels[1].A
#
# [solver2, work2] = asa_solver(A2, initial_targets=initial_targets,
#                               max_targets=max_targets,
#                               min_targets=min_targets,
#                               num_initial_targets=num_initial_targets,
#                               targets_iters=targets_iters, conv_tol=conv_tol,
#                               weak_tol=weak_tol, local_weak_tol=local_weak_tol,
#                               max_coarse=max_coarse, max_levels=max_levels,
#                               max_level_iterations=max_level_iterations,
#                               prepostsmoother=prepostsmoother,
#                               smooth=smooth, strength=strength, aggregate=aggregate,
#                               coarse_solver=coarse_solver, coarse_size=coarse_size)
# x02 = solver.levels[0].R*x0
# b2 = solver.levels[0].R * b
# xn2 = solver2.solve(b2, x0=x02, tol=tol, maxiter=maxiter, callback=lambda x: callback(A2,x))
#
# print "\nFinal error energy: ||xn|| = %1.5e\n"%(A_norm(xn2, A2))

sys.stdout.flush()

###############################################################################
# Plots
###############################################################################

# Plot the final bad guy
if plot_targets:
    fig = pyplot.figure('Final Error')
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Y, X, xn.reshape((n,n)), cmap=cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=5)


    # Plot targets
    fig = pyplot.figure("Target #0")
    ax = fig.gca(projection='3d')
    B = solver.levels[0].history['B'][0]
    surf = ax.plot_surface(Y, X, B[:,0].reshape((n,n)), cmap=cm.coolwarm)
    def data_gen(i):
        ax.clear()
        B = solver.levels[0].history['B'][i]
        surf = ax.plot_surface(Y, X, B[:,0].reshape((n,n)), cmap=cm.coolwarm)
    pam_ani = animation.FuncAnimation(fig, data_gen,
                    numpy.arange(0, len(solver.levels[0].history['B'])),
                    interval=600, repeat=True, blit=False)
    # for (h, B) in enumerate(solver.levels[0].history['B']):
    #     for i in range(B.shape[1]):
    #         fig = pyplot.figure("Target #%d, iter %d"%(i, h))
    #         ax = fig.gca(projection='3d')
    #         surf = ax.plot_surface(Y, X, B[:,i].reshape((n,n)), cmap=cm.coolwarm)
    #         fig.colorbar(surf, shrink=0.5, aspect=5)
    pyplot.show()

print "done"

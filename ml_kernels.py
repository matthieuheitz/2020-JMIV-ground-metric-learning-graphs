#!/usr/bin/env python
"""
ml_kernels.py

Kernel functions for Sinkhorn

"""

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import os
# if "CHOLMOD_USE_GPU" in os.environ:
#     print("CHOLMOD_USE_GPU = ",os.environ["CHOLMOD_USE_GPU"])
# else:
#     print("CHOLMOD_USE_GPU not in environment")
# os.environ["CHOLMOD_USE_GPU"] = "1"
# print("CHOLMOD_USE_GPU = ",os.environ["CHOLMOD_USE_GPU"])
try:
    import sksparse.cholmod
except ModuleNotFoundError as e:
    print("sksparse.cholmod could not be found. It is optional")

from ml_parameters import *
import np_utils as npu
import ml_core as mlc

# Debug
import cProfile



# New kernel that does all substeps inside, but have M and Mpre as global variables.
# Also supports the Crank-Nicolson numerical scheme
# This is the latest kernel.
class NumpyKernelFunctionNew3(torch.autograd.Function):

    @staticmethod
    def precomputation(weights, extra_prm):

        n = extra_prm['n']
        N = extra_prm['N']
        dim = extra_prm['dim']
        t = extra_prm['t_heat']
        K = extra_prm['k_heat']
        alpha = extra_prm['alpha']

        # Convert to numpy
        wn = prm.tensor2npy(weights)

        # Create the sparse weighted adjaceny matrix
        Ws = npu.get_grid_weighted_adjacency_npsparse(n,dim,wn)

        # Build the Identity matrix
        Id = scipy.sparse.eye(N)

        # Build the p.s.d. Laplacian matrix
        L = (scipy.sparse.diags(np.array(np.sum(Ws, 1).T), [0]) - Ws)  # Neumann boundary conditions
        L = L*alpha

        # By default, use Backward Euler
        if 'numerical_scheme' not in extra_prm: extra_prm['numerical_scheme'] = "backward_euler"

        # Build the matrix (Id+t*L) (not Id-t*L, because we built a positive definite Laplacian
        if extra_prm['numerical_scheme'] == "backward_euler":
            M = Id + (t/K) * L
        elif extra_prm['numerical_scheme'] == "crank_nicolson":
            M = Id + (t/(2*K)) * L
            mySparseSolver.Mcn = 2*scipy.sparse.eye(N) - M
        else:
            print("Error: Unrecognized numerical scheme: '%s'"%extra_prm['numerical_scheme'])
            exit(-1)

        if prm.solver_type == "SparseDirect":
            # Direct : Pre-factorization :
            if extra_prm['SD_algo'] == "LU":
                Mpre = scipy.sparse.linalg.splu(M.tocsc(), permc_spec="MMD_AT_PLUS_A")
                mySparseSolver.solve = Mpre.solve
            elif extra_prm['SD_algo'] == "Cholesky":
                Mpre = sksparse.cholmod.cholesky(M.tocsc(), ordering_method="metis")     # ordering_method: “natural”, “amd”, “metis”, “nesdis”, “colamd”, “default” and “best”.
                # Mpre = sksparse.cholmod.cholesky((t/K*L).tocsc(), beta=1, ordering_method="metis")    # Isn't faster
                mySparseSolver.solve = Mpre.solve_A
            else:
                print("SparseDirect solver unrecognized")
                exit(-1)

        else:
            raise ValueError("Solver type '%s' not recognized"%prm.solver_type)

        mySparseSolver.Mpre = Mpre
        mySparseSolver.M = M


    @staticmethod
    def forward(ctx, weights, b, extra_prm):

        # Get parameters
        N = extra_prm['N']
        K = extra_prm['k_heat']

        # If we use Scipy to solve the system
        if prm.solver_type == "SparseDirect":
            b = prm.tensor2npy(b).squeeze()
            # Block RHS
            if b.ndim == 2:
                N,S = b.shape
                # Conjugate Gradient, K times
                u_l = np.zeros([K+1,N,S])
                # u_l = np.zeros([N,K+1])
            elif b.ndim == 1:
                u_l = np.zeros([K+1,N])

            u_l[0] = b

            # t0 = time.time()
            # Solve with the chosen method
            for l in range(1, K + 1):
                if extra_prm['numerical_scheme'] == "backward_euler":
                    u_l[l] = mySparseSolver.solve(u_l[l-1])
                elif extra_prm['numerical_scheme'] == "crank_nicolson":
                    u_l[l] = mySparseSolver.solve(mySparseSolver.Mcn.dot(u_l[l-1]))

            # Save data for backward
            ctx.save_for_backward(prm.from_numpy(u_l), weights)   # Doesn't work for non torch tensors arguments
            ctx.intermediate = extra_prm

            return prm.from_numpy(u_l[K])

        else:
            raise ValueError("Solver type '%s' not recognized" % prm.solver_type)

    @staticmethod
    def backward(ctx, grad_output):

        # import pdb
        # pdb.set_trace()

        # TODO: This can potentially save quite a bit of time because I think it's longer to solve systems where the RHS is ridiculously small.
        #  But it does mean less precise gradients after, and might cause errors in LBFGS.
        # # If grad is too small
        # if grad_output.norm() < 1e-100:
        #     print("Grad norm is too small (<1e-100)")
        #     del ctx.intermediate
        #     return None, None, None, None, None

        # pr = cProfile.Profile()
        # pr.enable()

        # ctx.saved_tensors doesn't support non-tensor variables, so we use ctx.intermediate.
        extra_prm = ctx.intermediate
        u_l, weights = ctx.saved_tensors
        u_l = prm.tensor2npy(u_l)

        # Set initial grads
        grad_weights = grad_rhs = None

        # Get parameters
        n = extra_prm['n']
        dim = extra_prm['dim']
        K = extra_prm['k_heat']
        t = extra_prm['t_heat']
        alpha = extra_prm['alpha']

        # If we use Scipy to solve the system
        if prm.solver_type == "SparseDirect":

            v_l = np.zeros_like(u_l)
            # v_l = np.zeros([N,K+1]) # TODO : Store them so that they can be read continuously across the K, check if faster

            u = v = c = None
            if extra_prm['numerical_scheme'] == "backward_euler":

                v_l[K] = prm.tensor2npy(grad_output)

                # t0 = time.time()
                for l in np.arange(K-1, -1, -1):
                    v_l[l] = mySparseSolver.solve(v_l[l+1])
                # print("%0.2e,%f"%(grad_output.norm(),time.time()-t0))

                # Keep vectors of interest and compute laplacian factor c
                u = u_l[1:]
                c = alpha*t/K
                v = v_l[:-1]

            elif extra_prm['numerical_scheme'] == "crank_nicolson":

                v_l[K] = prm.tensor2npy(grad_output)
                v_l[K-1] = mySparseSolver.solve(v_l[K])

                # t0 = time.time()
                for l in np.arange(K-2, -1, -1):
                    v_l[l] = mySparseSolver.solve(mySparseSolver.Mcn.dot(v_l[l+1]))
                # print("%0.2e,%f"%(grad_output.norm(),time.time()-t0))

                u = u_l[1:]+u_l[:-1]
                c = alpha*t/(2*K)
                v = v_l[:-1]

            # Compute gradient
            # t0 = time.time()
            A = prm.tensor2npy(weights)
            grad_A = np.zeros_like(A)

            # Formula for each p: dz/dA(p) = -c*(v_i - v_{i+n**p})*(u_i - u_{i+n**p})
            for p in range(dim):
                npp = n**p
                hA1 = u.copy()               # u_i
                hA1[:,:-npp] -= u[:,npp:]    # -u_{i+n**p}
                hA2 = v.copy()               # v_i
                hA2[:,:-npp] -= v[:,npp:]    # -v_{i+n**p}
                grad_A[p] = -c*(np.sum(hA1*hA2,0))


            if 'gradcheck' in extra_prm and extra_prm['gradcheck'] == True:
                # Correct gradient for comparison with gradcheck
                # gradcheck computes a finite difference on all elements of the input,
                # even the zeros that are there for convenience of the input size,
                # but that do not represent real edges of the graph (they would be the ones making periodic boundaries)
                for p in range(dim):
                    grad_A[p,-n**p:] = 0
            else:
                # Apply mask to remove gradients w.r.t zeros of the input.
                grad_A *= (A>0)

            # print("New version:", time.time() - t0)
            # print("Max abs err:",torch.max(grad_A-grad).item())  # Always ~= 1e-16

        # Return gradient only if needed
        if ctx.needs_input_grad[0]:
            grad_weights = prm.from_numpy(grad_A)

        if ctx.needs_input_grad[1]:
            if extra_prm['numerical_scheme'] == "backward_euler":
                grad_rhs = prm.from_numpy(v[0])
            elif extra_prm['numerical_scheme'] == "crank_nicolson":
                grad_rhs = prm.from_numpy(mySparseSolver.Mcn.dot(v[0]))

        # This deletes what we stored in ctx. If we don't do that, these objects are not freed, and memory grows and grows.
        # We could have used ctx.save_for_backward(), but it only takes arguments that are torch.tensor...
        # The only problem with this, is that doing .backward(retain_graph=True) doesn't work, because ctx.intermediate
        # gets deleted after the first backward, and I have no access to the variable retain_graph in this function, to
        # only delete when retain_graph=False
        # if 'gradcheck' not in extra_prm or not extra_prm['gradcheck']:
        #     del ctx.intermediate

        return grad_weights, grad_rhs, None


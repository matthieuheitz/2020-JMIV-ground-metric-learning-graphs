#!/usr/bin/env python
"""
ml_parameters.py

Metric Learning Settings

"""

import os
import sys
import torch
import numpy as np

__author__ = "Matthieu Heitz"


# Parameter class
class OTMLParameters:

    # By default, tensors are float64, on CPU
    def __init__(self):
        self.outdir = "../ProgOutput/yolo"   # Folder where to save all data
        self.dtype = torch.float64           # The Pytorch Tensor type
        self.device = torch.device('cpu')    # The device where tensors live
        self.clone2npy = lambda x : x.detach().clone().numpy()  # Result doesn't share the underlying data with x
        self.tensor2npy = lambda x : x.detach().numpy()         # Result shares the underlying data with x
        self.cuda_enabled = False            # Do all computations on the GPU with CUDA tensors
        self.numba_jit_enabled = False           # Use JIT compiled functions when possible (use only with CPU (cuda_enabled = False))
        self.plot_loss = True                # Save a plot of the loss evolution
        self.show_plot = False               # Show the plot of the loss evolution in a matplotlib window
        self.save_frequency = 1              # Save every n iterations
        self.save_graph_plot = False         # Flag to save the adjacency matrix as a colored graph
        self.iter_save = False               # Flag to save data at every iteration
        self.iter_num = 0                    # Iteration counter
        self.apsp_algo = 'Numpy_kernel'      # Name of the APSP algorithm to compute the distance (and cost) matrix.
        self.solver_type = "SparseDirect"    # Type of the linear solver
        self.metric_regul = "metric_noregul" # Regularization of the metric
        self.varch_f = lambda x : x          # Variable change: Forward - from normal domain to other domain
        self.varch_b = lambda x : x          # Variable change: Backward - from other domain to normal domain

        # Type of metric
        # "grid_edges_scalar": first code with one value per edge
        # "grid_vertices_tensor_diag": second code with dim weights per vertex (2 for 2D, 3 for 3D): equivalent to diagonal tensors
        # Others are not coded yet. For example:
        # "graph_edges_scalar": for arbitrary graph, with one value per edge
        # "grid_vertices_tensor_full": grid with one full tensor at each vertex
        self.metric_type = "grid_vertices_tensor_diag"

    # Always copies data
    def tensor(self, data, requires_grad=False, dtype=None):
        datatype = dtype or self.dtype
        # To avoid UserWarning: "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or
        # sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)"
        if isinstance(data, torch.Tensor):
            return data.clone().detach().requires_grad_(requires_grad).type(datatype)
        else:
            return torch.tensor(data, dtype=datatype, device=self.device, requires_grad=requires_grad) # Maybe faster ?
            # return torch.tensor(data).type(self.dtype).to(self.device).requires_grad_(requires_grad)

    # Avoids copy of ndarray, will share the same underlying data
    def from_numpy(self, ndarray, requires_grad=False):
        return torch.from_numpy(ndarray).type(self.dtype).to(self.device).requires_grad_(requires_grad)
        # can't do that because "from_numpy() takes no keyword arguments"
        # return torch.from_numpy(ndarray, dtype=self.dtype, device=self.device, requires_grad=requires_grad)

    def enable_cuda(self, id=0):
        if torch.cuda.is_available():
            if id >= torch.cuda.device_count():
                print("Error: Invalid Device ID. Defaulting to CPU")
                self.enable_cpu()
                return
            self.device = torch.device("cuda:%d"%id)
            self.cuda_enabled = True
            self.clone2npy = lambda x: x.cpu().detach().clone().numpy()
            self.tensor2npy = lambda x: x.cpu().detach().numpy()
        else:
            print("Error: CUDA is not available. Defaulting to CPU")
            self.enable_cpu()

    def enable_cpu(self):
        self.device = torch.device("cpu")
        self.cuda_enabled = False
        self.clone2npy = lambda x : x.detach().clone().numpy()
        self.tensor2npy = lambda x : x.detach().numpy()

    # Function that makes old sets of parameters compatible with the current code version.
    def forward_compatibility_prm_dict(self, prm_dict):
        # Make a copy of the dict (careful, this is only a shallow copy, so no nested references allowed)
        new_dict = prm_dict.copy()

        if 'x_on_points' in new_dict:
            if new_dict['x_on_points']: new_dict['metric_type'] = "grid_vertices_tensor_diag"
            else:                       new_dict['metric_type'] = "grid_edges_scalar"
            del new_dict['x_on_points']
        if 'metric_type' not in new_dict:   # If there was neither x_on_points nor metric_type, it means it's the first version of the code.
            new_dict['metric_type'] = "grid_edges_scalar"
        if 'dim' not in new_dict:
            new_dict['dim'] = int(round(np.log(new_dict['N'])/np.log(new_dict['n'])))
        if 'alpha' not in new_dict:
            h = 1/(new_dict['n']-1)
            new_dict['alpha'] = 1/(h**2)
        if 'metric_regul_ro' not in new_dict:
            new_dict['metric_regul_ro'] = 0
        if 'metric_regul_ro_lap' not in new_dict:
            new_dict['metric_regul_ro_lap'] = 0
        if 'm_t_heat' not in new_dict:
            new_dict['m_t_heat'] = 0
        if 'colorspace' not in new_dict:
            new_dict['colorspace'] = "RGB"

        return new_dict


prm = OTMLParameters()


# Logger that duplicates output to terminal and to file
# Not portable on Windows though...
class Logger(object):

    def __init__(self,logfile):
        import warnings
        warnings.filterwarnings("default")

        # Works but we loose colors in the terminal
        import subprocess
        self.tee = subprocess.Popen(["tee", logfile], stdin=subprocess.PIPE)
        os.dup2(self.tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(self.tee.stdin.fileno(), sys.stderr.fileno())

    def __del__(self):
        self.tee.stdin.close()
        sys.stdout.close()
        sys.stderr.close()


# Class that holds the solver objects.
# This way, I don't have to pass them in the autograd custom functions.
class MySparseSolver:

    def __init__(self):
        self.W = None           # Sparse weighted adjacency matrix (swam)
        self.Windices = None    # Indices of non-zeros in W
        self.M = None           # Sparse implicit operator for smoothing: M = Id - t/K*L
        self.Mpre = None        # Pre-factorization of the matrix M
        self.Mcn = None         # Sparse explicit operator for smoothing, when using Crank-Nicolson
        self.solve = None       # Reference to Mpre.solve() (LU: Mpre.solve(), Cholesky: Mpre.solve_A())


# Global object, accessible from everywhere
mySparseSolver = MySparseSolver()


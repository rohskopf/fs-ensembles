import sys
import numpy as np
import pickle
import torch
from pathlib import Path
import glob
from fitsnap3lib.fitsnap import FitSnap
import pickle
import time

list_of_weights=sorted(glob.glob('fit*/Ta_Pytorch.pt'))
print('These are the weights of the nn',list_of_weights)

print('Loading config file')

t0 = time.time()

with open(r"configs.pickle", "rb") as file:
    configs = pickle.load(file)

t1 = time.time()
total = t1-t0

print('Total time (s) to load config file',total)

# Declare number of fits.
nfits = len(list_of_weights)

# Create base dictionary of settings, will be modified for each fit.
# NOTE: Use same settings as used to train model, but use "save_state_input": "Ni_Pytorch.pt"
settings = \
{
"BISPECTRUM":
    {
    "numTypes": 1,
    "twojmax": 6,
    "rcutfac": 4.67637,
    "rfac0": 0.99363,
    "rmin0": 0.0,
    "wj": 1.0,
    "radelem": 0.5,
    "type": "Ta",
    "wselfallflag": 0,
    "chemflag": 0,
    "bzeroflag": 1,
    "bikflag": 1,
    "dgradflag": 1
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1,
    "force": 1,
    "per_atom_energy": 1,
    "nonlinear": 1
    },
"PYTORCH":
    {
    "layer_sizes": "num_desc 64 64 1",
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "batch_size": 4, # 363 configs in entire set
    "save_state_input": "Ta_Pytorch.pt"
    },
"SOLVER":
    {
    "solver": "PYTORCH"
    }
}

# Make a list of settings for each fit.

from copy import deepcopy
settings_lst = [deepcopy(settings) for _ in range(nfits)]
for i,s in enumerate(settings_lst):
    s['PYTORCH']['save_state_input'] = list_of_weights[i]
print(settings_lst)

#from mpi4py import MPI
#comm = MPI.COMM_WORLD

# Load pytorch file from a previous fit.
instances = [FitSnap(settings, arglist=["--overwrite"]) for _ in range(nfits)]
for i, inst in enumerate(instances):
    t0 = time.time()
    inst.solver.configs = configs
    (energies_model, forces_model) = inst.solver.evaluate_configs(config_idx=None, standardize_bool=True)
    t1 = time.time()
    total = t1-t0
    print('Total time (s) to obtain forces and energies from config file',total)

    with open("energies_model_"+str(i+1).zfill(2), "wb") as fp:   #Pickling
        pickle.dump(energies_model, fp)
        
    with open("forces_model_"+str(i+1).zfill(2), "wb") as fp:   #Pickling
        pickle.dump(forces_model, fp)

    # Delete the instance to free memory.
    del inst

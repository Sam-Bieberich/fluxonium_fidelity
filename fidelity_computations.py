import qutip as qt
import numpy as np
import scqubits as scq
import matplotlib.pyplot as plt
import itertools
import warnings
import os
import time

import import_functions  # local file with helper functions


levels = 6
fluxonium = scq.Fluxonium(EJ=8.9, EC=2.5, EL=0.5, flux=0.48, cutoff=110)
c_ops = None  # will be initialized once below

def init_c_ops():
    gamma_ij = {}
    for j in range(1, levels):
        for i in range(j):
            t1 = fluxonium.t1_capacitive(j, i, Q_cap=1e5, total=False)
            if t1 is not None and t1 > 0:
                rate = 1.0 / t1
                gamma_ij[(i, j)] = rate
                gamma_ij[(j, i)] = rate
    c_ops_local = []
    for (i, j), gamma in gamma_ij.items():
        cop = np.sqrt(gamma) * qt.basis(levels, i) * qt.basis(levels, j).dag()
        c_ops_local.append(cop)
    return c_ops_local

scq.settings.T1_DEFAULT_WARNING=False

if c_ops is None:
    c_ops = init_c_ops()

evals, evecs = fluxonium.eigensys(evals_count=levels)
n_op_energy_basis = qt.Qobj(fluxonium.process_op(fluxonium.n_operator(), energy_esys=(evals, evecs)))
H0 = qt.Qobj(np.diag(evals)) * 2 * np.pi
A = 0.4 * 2 * np.pi
drive_op = n_op_energy_basis
H = [H0, [A * drive_op, 'cos(wd * t)']]

########################################### Running ###########################################
# Note that the parameters below are chosen to be near optimal values
# for a pi-pulse around the x-axis (X gate). Values shown in the example below were
# used for plot generation in the poster. 


evals, _ = fluxonium.eigensys(evals_count=levels)
omega_d_base = (evals[1] - evals[0]) * 2 * np.pi

print("omega d is", omega_d_base)

dimensions_omega = 3
dimensions_time = 6

omega_d_array = np.linspace(omega_d_base - 0.01, omega_d_base + 0.01, dimensions_omega)

t_g_array = np.linspace(24.2, 24.6, dimensions_time)
# param_pairs = list(itertools.product(omega_d_array, t_g_array))

# print(f"Total simulations to run: {len(param_pairs)}")

warnings.filterwarnings(
    "ignore",
    module="qutip.*"  # Regex pattern to match all warnings from qutip
)
scq.settings.T1_DEFAULT_WARNING=False

try:
    import pathos.multiprocessing as mp
except ImportError:
    print(
        "using std lib version of multiprocessing; consider installing pathos; it's much more robust"
    )
    import multiprocessing as mp

def _default_kwargs():
    return {"num_cpus": os.cpu_count() or 1}



#single process
# fidelity_results = param_map(evolve, [omega_d_array, t_g_array])

#parallel process
fidelity_results = param_map(evolve, [omega_d_array, t_g_array], map_fun=parallel_map_qutip)


########################################### Prints ###########################################

print("t g array is ", t_g_array)
print("omega d array is ", omega_d_array)
print("Omega by time dimensions are ", dimensions_omega, " by ", dimensions_time)
print("fidelity results are ", fidelity_results)

fidelity_results = np.asarray(fidelity_results,dtype = float)

density(1 - fidelity_results, omega_d_array, t_g_array, title="Average X Gate Error", x_label = "Drive frequency (GHz)", y_label = "Gate Time (ns)", data_label = "Error")
# Imports

import qutip as qt
import numpy as np
import scqubits as scq
import matplotlib.pyplot as plt
import itertools
import warnings
import os
import time
import multiprocessing as mp


# function for plotting the density (heatmap) of a 2D array
def density(
    z,
    x_array,
    y_array,
    x_label=None,
    y_label=None,
    data_label=None,
    title=None,
    fig_ax=None,
    process_fun=None,
    z_min_max=None,
    norm_fun=None,
    show_values=False,
    **kwargs,
):
    """
    norm_fun should typically be: lambda (z_min, z_max): SymLogNorm(1e-9, 1.0, vmin=z_min, vmax=z_max)
    with: from matplotlib.colors import LogNorm, SymLogNorm
    """
    global global_fig, global_axes
 
    fig, ax = fig_ax or plt.subplots(1, 1, 
            # figsize=(14,12)
            )
    global_fig, global_axes = fig, ax
 
    x, y = np.meshgrid(x_array, y_array)
 
    z = np.ma.array(z)
 
    if process_fun is not None:
        z = process_fun(z)
 
    if z_min_max is not None and not callable(z_min_max):
        z_min = np.nanmin(z) if z_min_max[0] is None else z_min_max[0]
        z_max = np.nanmax(z) if z_min_max[1] is None else z_min_max[1]
    else:
        z_min, z_max = np.nanmin(z), np.nanmax(z)
 
    print(z_min, z_max)
 
    norm = None
    if norm_fun is not None:
        norm = norm_fun(z_min, z_max)
        z_min, z_max = None, None
 
    # extent=[x_array[0], x_array[-1], y_array[0], y_array[-1]]
    # im = ax.imshow(z.T,
    # interpolation=None,
    # cmap="jet",
    # origin='lower',
    # vmin=z_min, vmax=z_max,
    # # extent=extent,
    # aspect='auto',
    # **kwargs
    # )
 
    im = ax.pcolormesh(
        x_array,
        y_array,
        z.T,
        cmap="jet",
        vmin=z_min,
        vmax=z_max,
        norm=norm,
        shading="auto",
        **kwargs,
    )
 
    c1 = fig.colorbar(im, ax=ax)
    c1.ax.set_title(data_label, fontsize=12)
 
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
 
    ax.set_title(title)
    fig.canvas.draw()

 
    return fig, ax


def evolve(omega_d, t_g):
    args = {'wd': omega_d}
    print("working")
    options = qt.Options(nsteps=10000000, store_states=True, atol=1e-12, rtol=1e-11)
    propagator = qt.propagator(H, t_g, args=args, options=options)
    propagator_kraus = qt.to_kraus(propagator)
    propagator_2x2 = [qt.Qobj(k.full()[:2, :2]) for k in propagator_kraus]
    p_2x2_super = qt.kraus_to_super(propagator_2x2)
    fidelity = qt.average_gate_fidelity(p_2x2_super, qt.sigmax())
    return fidelity    


def evolve_wrapped(params):
    omega_d, t_g = params
    return evolve(omega_d, t_g)

#note that init_cops is defined in the fidelity_computations.py file directly to avoid circular imports

def parallel_map_qutip(task, values, task_args=tuple(), task_kwargs={}, **kwargs):
    """
    ---
    peterg NOTE: This is a modified parallel_map taken from qutip's source
    code. The version I would typically use, can break on some code that uses
    qutip's internal routines. This is likely due to some conflict with the
    openmp code (which the code below "turns off" via an environmental
    variable).  My slight modification is to to allow pathos (assumed imported
    as "mp") module to be used, as it handles pickle'ing various fancy objects
    much better.
    ---

    Parallel execution of a mapping of `values` to the function `task`. This
    is functionally equivalent to::

 

        result = [task(value, *task_args, **task_kwargs) for value in values]

    Parameters
    ----------
    task : a Python function
        The function that is to be called for each value in ``task_vec``.
    values : array / list
        The list or array of values for which the ``task`` function is to be
        evaluated.
    task_args : list / dictionary
        The optional additional argument to the ``task`` function.
    task_kwargs : list / dictionary
        The optional additional keyword argument to the ``task`` function.
    progress_bar : ProgressBar
        Progress bar class instance for showing progress.

    Returns
    --------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``.
    """
    os.environ["QUTIP_IN_PARALLEL"] = "TRUE"
    kw = _default_kwargs()
    if "num_cpus" in kwargs:
        kw["num_cpus"] = kwargs["num_cpus"]

    try:
        progress_bar = kwargs["progress_bar"]
        if progress_bar is True:
            progress_bar = TextProgressBar()
    except:
        progress_bar = BaseProgressBar()

 
    progress_bar.start(len(values))
    nfinished = [0] 

    def _update_progress_bar(x):
        nfinished[0] += 1
        progress_bar.update(nfinished[0])

    try:
        pool = mp.Pool(processes=kw["num_cpus"])
        async_res = [
            pool.apply_async(
                task, (value,) + task_args, task_kwargs, _update_progress_bar
            )
            for value in values
        ]
        while not all([ar.ready() for ar in async_res]):
            for ar in async_res:
                ar.wait(timeout=0.1)
        pool.terminate()
        pool.join()

    except KeyboardInterrupt as e:
        os.environ["QUTIP_IN_PARALLEL"] = "FALSE"
        pool.terminate()
        pool.join()
        raise e

    progress_bar.finished()
    os.environ["QUTIP_IN_PARALLEL"] = "FALSE"
    return [ar.get() for ar in async_res]


def parallel_map_adapter(f, iterable):
    return parallel_map_qutip(f, list(iterable))


def param_map(f, parameters, map_fun=map, dtype=object):

    dims_list = [len(i) for i in parameters]
    total_dim = np.prod(dims_list)
    parameters_prod = tuple(itertools.product(*parameters))

    data = np.empty(total_dim, dtype=dtype)
    # for i, d in enumerate(map_fun(f, parameters_prod)):
    #     data[i] = d
    for i, d in enumerate(map_fun(lambda args: f(*args), parameters_prod)):
        data[i] = d
    print("complete param map")
    return np.reshape(data, dims_list)


def varg_opt(data, axis=None, opt_fun=np.nanargmin):
    """
    Return an index of a (possibly) multi-dimensional array of the element that
    optimizes a given function along with the optimal value.
    """
    index = arg_opt(data, axis=axis, opt_fun=opt_fun)
    return index, data[index]
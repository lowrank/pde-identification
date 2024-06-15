import time

import matplotlib.pyplot as plt
import numpy as np

from models.evolution_diff_eq import EvolutionDiffEq

"""
Tests of a list of evolution differential equations.

Note that the x-axis is fixed as [0, 2 * pi], common initial condition.
"""
# models
models = {
    'KdV-Burgers': EvolutionDiffEq('0.5 * D[D[u]] - u * D[u] - (1.0/6) * D[D[D[u]]]'),
    'Kuramoto-Sivashinsky': EvolutionDiffEq('- D[D[u]] - 0.05 * D[D[D[D[u]]]] - 0.5 * D[u] * D[u]'),
    'Ginzburg-Landau': EvolutionDiffEq('D[D[u]] - u*u*u + u'),
    'FitzHugh–Nagumo': EvolutionDiffEq('D[D[u]] + u * u  - u*u*u'),
    'Fisher-KPP': EvolutionDiffEq('D[D[u]] + u - u*u'),
    'Zeldovich–Frank-Kamenetskii': EvolutionDiffEq('D[D[u]] + 0.5 * 20^2 * Exp[-20] * (u - u * u)  * Exp[20 * u]'),
    'Sine-Gordon': EvolutionDiffEq('0.25 * D[D[u]] + Sin[Sqrt[8*Pi] * u]'),
    'Swift-Hohenberg': EvolutionDiffEq('u - u * u * u - (2 * D[D[u]] + D[D[D[D[u]]]])'),
    'Eikonal': EvolutionDiffEq('0.25 * D[D[u]] + Abs[D[u]]'),
    'Porous Media': EvolutionDiffEq(' 3 * u * u *  D[D[u]] + 6 * u * D[u] * D[u]'),
    'Nonlinear Schrodinger': EvolutionDiffEq('-0.5 * D[D[v]] + v*(u*u+v*v)',
                                             '0.5 * D[D[u]]-(u*u+v*v)*u'),
    'KdV super': EvolutionDiffEq('0.25 * D[D[u]] - u*D[u] - (1.0/6)*D[D[D[u]]] + 0.5 * v*D[D[v]]',
                                 '0.25 * D[D[v]] + 0.5 * D[u]*v + u * D[v] - (4.0/6) *D[D[D[v]]]')
}

# Initial condition and domain parameters
dt, T = 0.001, 1
t_span = np.arange(dt, T + dt, dt)
x_axis = np.linspace(0, 2 * np.pi, 513)[0:-1]
x_size = len(x_axis)
y0 = 0.6 + 0.3 * np.sin(x_axis) + 0.15 * np.sin(2 * x_axis) + 0.1 * np.cos(3 * x_axis) + 0.02 * np.cos(4 * x_axis)

# plots
row = int(np.sqrt(len(models)))
col = int(np.ceil(len(models) / row))
fig, axs = plt.subplots(row, col, sharex=True)
fig.set_figheight(4 * row)
fig.set_figwidth(4 * col)

for key_order, (model_name, model) in enumerate(models.items()):
    print(f"-- model formula:", model_name, model)

    tic = time.time()
    model.wrap(model_name)  # compile model, locate lib in models/.tmp
    print(f"-- compiling time: {(time.time() - tic) * 1000:.1f} ms")

    if model.complex:
        tic = time.time()
        y = model.solve(np.concatenate((y0, np.zeros_like(y0))), t_span, rtol=1e-6, atol=1e-6)
        print(f"-- solving time: {(time.time() - tic) * 1000:.1f} ms")
    else:
        tic = time.time()
        y = model.solve(y0, t_span, rtol=1e-6, atol=1e-6)
        print(f"-- solving time: {(time.time() - tic) * 1000:.1f} ms")

    for i in range(5):
        if row > 1 and col > 1:
            axs[key_order // col][np.mod(key_order, col)].plot(x_axis, y[i * 200, 0:x_size])
        else:
            axs[key_order].plot(x_axis, y[i * 200, 0:x_size])
    if row > 1 and col > 1:
        axs[key_order // col][np.mod(key_order, col)].set_title(f'{model_name}', size=14)
    else:
        axs[key_order].set_title(f'{model_name}', size=14)
plt.show()

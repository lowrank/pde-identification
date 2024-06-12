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
    'Sine-Gordon': EvolutionDiffEq('D[D[u]] + 2 * Sin[Sqrt[8*Pi] * u]'),
    'Swift-Hohenberg': EvolutionDiffEq('u - (2 * D[D[u]] + D[D[D[D[u]]]])'),
    'Ekonal': EvolutionDiffEq('0.01 * D[D[u]] + Abs[D[u]]')
}

# Initial condition and domain parameters
dt, T = 0.001, 1
t_span = np.arange(dt, T + dt, dt)
x_axis = np.linspace(0, 2 * np.pi, 513)[0:-1]
y0 = 0.5 + 0.4 * np.sin(x_axis) + 0.2 * np.sin(2 * x_axis) + 0.04 * np.cos(3 * x_axis)

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

    tic = time.time()
    y = model.solve(y0, t_span, rtol=1e-6, atol=1e-6)
    print(f"-- solving time: {(time.time() - tic) * 1000:.1f} ms")

    for i in range(5):
        if row > 1 and col > 1:
            axs[key_order // row][np.mod(key_order, row)].plot(x_axis, y[i * 200, :])
        else:
            axs[key_order].plot(x_axis, y[i * 200, :])
    if row > 1 and col > 1:
        axs[key_order // row][np.mod(key_order, row)].set_title(f'{model_name}', size=14)
    else:
        axs[key_order].set_title(f'{model_name}', size=14)
plt.show()

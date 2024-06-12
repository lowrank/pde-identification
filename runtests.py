from models.evolution_diff_eq import EvolutionDiffEq
import numpy as np
import matplotlib.pyplot as plt
import time

"""
create an evolution type differential equation of

1. KdV (general) equation
2. KdV-Burgers equation
3. Kuramoto-Sivashinsky equation

Note that the x-axis is fixed as [0, 2 * pi]
"""

models = {
    'KdV': EvolutionDiffEq('- u * D[u] - (1.0/6) * D[D[D[u]]]'),
    'KdV-Burgers': EvolutionDiffEq('0.01 * D[D[u]] -2 * u * D[u] - 0.25 * D[D[D[u]]]'),
    'Kuramoto-Sivashinsky': EvolutionDiffEq('- D[D[u]] - 0.01 * D[D[D[D[u]]]] - 0.5 * D[u] * D[u]')
}

fig, axs = plt.subplots(len(models), 1, sharex=True)
fig.set_figheight(10)
fig.set_figwidth(6)

for key_order, (model_name, model) in enumerate(models.items()):
    print(f"-- model formula:", model_name, model)

    tic = time.time()
    model.wrap(model_name)  # compile model

    print(f"-- compiling time: {(time.time() - tic) * 1000:.1f} ms")

    dt, T = 0.001, 1
    t_span = np.arange(dt, T + dt, dt)
    x_axis = np.linspace(0, 2 * np.pi, 513)[0:-1]

    y0 = np.sin(x_axis)

    tic = time.time()
    y = model.solve(y0, t_span, rtol=1e-6, atol=1e-6)
    print(f"-- solving time: {(time.time() - tic) * 1000:.1f} ms")

    for i in range(50):
        axs[key_order].plot(x_axis, y[i * 20, :])
    axs[key_order].title.set_text(f'Solutions to  {model_name}  equation')
plt.show()

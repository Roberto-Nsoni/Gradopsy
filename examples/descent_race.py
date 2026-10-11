"""
The `descent` demo, rendered with matplotlib. Ugly on purpose: this is here to see the race
move today, and to be the readable reference of what data the real frontend has to consume.

    python examples/descent_race.py --surface double_well
    python examples/descent_race.py --surface rosenbrock --save race.gif
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

from ml.data import SURFACES
from ml.optim import SGD, Momentum, RMSProp, Adam, BFGS, Newton
from ml.training import SurfaceRunner, Recorder

# Each optimizer needs its own step size to be worth watching: comparing SGD at lr=0.05 with
# BFGS at its own alpha is the honest comparison, not forcing everyone into the same number.
CORREDORES = [
    ("SGD",      SGD,      {"lr": 0.002}, "#e15759"),
    ("Momentum", Momentum, {"lr": 0.002}, "#f28e2b"),
    ("RMSProp",  RMSProp,  {"lr": 0.02},  "#edc948"),
    ("Adam",     Adam,     {"lr": 0.05},  "#59a14f"),
    ("BFGS",     BFGS,     {},            "#4e79a7"),
    ("Newton",   Newton,   {},            "#76b7b2"),
]


def correr(surface, steps, start):
    """One recording per optimizer, all of them from the same starting point."""
    grabaciones = {}

    for nombre, cls, kwargs, _ in CORREDORES:
        runner = SurfaceRunner(surface, cls, start=start, **kwargs)
        rec = Recorder()

        for _ in range(steps):
            rec.add(runner.step())
            if not np.isfinite(runner.position).all():
                break           # the unbounded surfaces blow up, and that is the honest result

        grabaciones[nombre] = rec

    return grabaciones


def animar(surface, grabaciones, steps, guardar=None):
    fig, (ax, ax_loss) = plt.subplots(1, 2, figsize=(13, 6), width_ratios=[3, 2])
    fig.suptitle(surface.name, fontsize=14)

    XX, YY, Z = surface.grid(n=300)
    norm = LogNorm(vmin=max(Z.min(), 1e-3), vmax=Z.max()) if surface.log_scale else None
    ax.contourf(XX, YY, Z, levels=40, norm=norm, cmap="Greys_r")
    ax.contour(XX, YY, Z, levels=20, norm=norm, colors="black", linewidths=0.3, alpha=0.4)

    for (mx, my) in surface.minima:
        ax.plot(mx, my, "*", color="white", markersize=16, markeredgecolor="black")
    if not surface.bounded:
        ax.set_title("unbounded: it has no minimum", fontsize=9)

    rastros, puntos, curvas = {}, {}, {}
    for nombre, _, _, color in CORREDORES:
        rastros[nombre], = ax.plot([], [], "-", color=color, linewidth=1.6, alpha=0.9, label=nombre)
        puntos[nombre], = ax.plot([], [], "o", color=color, markersize=7, markeredgecolor="black")
        curvas[nombre], = ax_loss.plot([], [], "-", color=color, linewidth=1.6, label=nombre)

    ax.set_xlim(surface.x_range); ax.set_ylim(surface.y_range)
    ax.legend(loc="upper right", fontsize=8)

    ax_loss.set_xlim(0, steps); ax_loss.set_xlabel("step"); ax_loss.set_ylabel("loss")
    if surface.bounded:
        ax_loss.set_yscale("symlog", linthresh=1e-6)
    ax_loss.grid(alpha=0.2)

    perdidas = {n: g.column("loss") for n, g in grabaciones.items()}
    caminos = {n: g.column("params") for n, g in grabaciones.items()}
    ax_loss.set_ylim(min(p.min() for p in perdidas.values()), max(p[0] for p in perdidas.values()))

    def frame(k):
        for nombre, _, _, _ in CORREDORES:
            camino = caminos[nombre][:k + 1]
            rastros[nombre].set_data(camino[:, 0], camino[:, 1])
            puntos[nombre].set_data(camino[-1:, 0], camino[-1:, 1])
            curvas[nombre].set_data(np.arange(len(perdidas[nombre][:k + 1])), perdidas[nombre][:k + 1])

        return list(rastros.values()) + list(puntos.values()) + list(curvas.values())

    anim = FuncAnimation(fig, frame, frames=steps, interval=30, blit=True, repeat=False)
    plt.tight_layout()

    if guardar:
        anim.save(guardar, writer="pillow", fps=30)
        print(f"guardado en {guardar}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", default="double_well", choices=sorted(SURFACES))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--start", type=float, nargs=2, default=None)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    surface = SURFACES[args.surface]
    grabaciones = correr(surface, args.steps, args.start)

    for nombre, _, _, _ in CORREDORES:
        rec = grabaciones[nombre]
        x, y = rec[-1]["params"]
        print(f"  {nombre:<9} {len(rec):>4} pasos -> ({x:+.4f}, {y:+.4f})  f={rec[-1]['loss']:+.6g}")

    animar(surface, grabaciones, min(args.steps, min(len(g) for g in grabaciones.values())), args.save)


if __name__ == "__main__":
    main()

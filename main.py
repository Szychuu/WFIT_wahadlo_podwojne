#!/usr/bin/env python3
import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

# Parametry fizyczne domyślne
g = 9.81


def prompt_initial_conditions():
    """
    Wyświetla okno z suwakami do ustawienia parametrów początkowych:
    θ1, θ2 (kąty początkowe), L1, L2 (długości), m1, m2 (masy).
    Zwraca wartości w tej kolejności.
    """
    # wartości inicjalne
    init_vals = {
        'theta1': 3 * np.pi / 7,
        'theta2': 3 * np.pi / 4,
        'L1': 1.0,
        'L2': 1.0,
        'm1': 1.0,
        'm2': 1.0
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.subplots_adjust(left=0.1, bottom=0.2)
    ax.set_axis_off()

    # suwaki
    s_theta1 = Slider(plt.axes([0.1, 0.80, 0.8, 0.04]), 'θ₁ [rad]', 0, 2 * np.pi, valinit=init_vals['theta1'])
    s_theta2 = Slider(plt.axes([0.1, 0.73, 0.8, 0.04]), 'θ₂ [rad]', 0, 2 * np.pi, valinit=init_vals['theta2'])
    s_L1 = Slider(plt.axes([0.1, 0.66, 0.8, 0.04]), 'L₁ [m]', 0.1, 5.0, valinit=init_vals['L1'])
    s_L2 = Slider(plt.axes([0.1, 0.59, 0.8, 0.04]), 'L₂ [m]', 0.1, 5.0, valinit=init_vals['L2'])
    s_m1 = Slider(plt.axes([0.1, 0.52, 0.8, 0.04]), 'm₁ [kg]', 0.1, 10.0, valinit=init_vals['m1'])
    s_m2 = Slider(plt.axes([0.1, 0.45, 0.8, 0.04]), 'm₂ [kg]', 0.1, 10.0, valinit=init_vals['m2'])

    btn = Button(plt.axes([0.4, 0.20, 0.2, 0.075]), 'Start')

    start_clicked = False

    def on_start(evt):
        nonlocal start_clicked
        start_clicked = True
        plt.close(fig)

    btn.on_clicked(on_start)

    while plt.fignum_exists(fig.number) and not start_clicked:
        plt.pause(0.1)

    if not start_clicked:
        return None

    return (
        s_theta1.val, s_theta2.val,
        s_L1.val, s_L2.val,
        s_m1.val, s_m2.val
    )


def equations(y, t, L1, L2, m1, m2):
    θ1, z1, θ2, z2 = y
    c, s = np.cos(θ1 - θ2), np.sin(θ1 - θ2)
    dz1 = (m2 * g * np.sin(θ2) * c
           - m2 * s * (L1 * z1 * z1 * c + L2 * z2 * z2)
           - (m1 + m2) * g * np.sin(θ1)) \
          / (L1 * (m1 + m2 * s * s))
    dz2 = ((m1 + m2) * (L1 * z1 * z1 * s
                        - g * np.sin(θ2) + g * np.sin(θ1) * c)
           + m2 * L2 * z2 * z2 * s * c) \
          / (L2 * (m1 + m2 * s * s))
    return [z1, dz1, z2, dz2]


def total_energy(Y, L1, L2, m1, m2):
    θ1, θ1d, θ2, θ2d = Y.T
    V = -(m1 + m2) * L1 * g * np.cos(θ1) - m2 * L2 * g * np.cos(θ2)
    T = 0.5 * m1 * (L1 * θ1d) ** 2 + 0.5 * m2 * ((L1 * θ1d) ** 2 + (L2 * θ2d) ** 2
                                                 + 2 * L1 * L2 * θ1d * θ2d * np.cos(θ1 - θ2))
    return T + V


def run_simulation(theta1_0, theta2_0, L1, L2, m1, m2, dt=0.01, tmax=45.0, playback_fps=60):
    t = np.arange(0, tmax + dt, dt)
    y0 = [theta1_0, 0.0, theta2_0, 0.0]
    Y = odeint(equations, y0, t, args=(L1, L2, m1, m2))

    # kontrola energii
    E0 = total_energy(np.array(y0).reshape(1, 4), L1, L2, m1, m2)
    if np.max(np.abs(total_energy(Y, L1, L2, m1, m2) - E0)) > 0.05:
        sys.exit("Drift energii za duży")

    θ1, θ2 = Y[:, 0], Y[:, 2]
    x1 = L1 * np.sin(θ1)
    y1 = -L1 * np.cos(θ1)
    x2 = x1 + L2 * np.sin(θ2)
    y2 = y1 - L2 * np.cos(θ2)

    n_frames = int(tmax * playback_fps)
    idx = np.linspace(0, len(t) - 1, n_frames).astype(int)
    return x1, y1, x2, y2, idx, L1, L2, playback_fps


def animate(x1, y1, x2, y2, idx, L1, L2, playback_fps, trail_length=500):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-L1 - L2 - 0.1, L1 + L2 + 0.1)
    ax.set_ylim(-L1 - L2 - 0.1, L1 + L2 + 0.1)
    ax.set_aspect('equal')
    plt.axis('off')

    rod_line, = ax.plot([], [], lw=2, color='black')  # Grubsza linia wahadła
    trail_line, = ax.plot([], [], lw=1, color='red', alpha=0.7)  # Czerwony ślad

    bob1 = Circle((0, 0), 0.05, fc='b', zorder=10)
    bob2 = Circle((0, 0), 0.05, fc='r', zorder=10)
    ax.add_patch(bob1)
    ax.add_patch(bob2)

    # Inicjalizacja danych śladu
    trail_x = []
    trail_y = []

    def init():
        rod_line.set_data([], [])
        trail_line.set_data([], [])
        bob1.center = (0, 0)
        bob2.center = (0, 0)
        return rod_line, trail_line, bob1, bob2

    def update(frame):
        nonlocal trail_x, trail_y

        i = idx[frame]
        rod_line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        bob1.center = (x1[i], y1[i])
        bob2.center = (x2[i], y2[i])

        # Aktualizacja śladu
        trail_x.append(x2[i])
        trail_y.append(y2[i])

        # Ogranicz długość śladu
        if len(trail_x) > trail_length:
            trail_x = trail_x[-trail_length:]
            trail_y = trail_y[-trail_length:]

        # Aktualizacja wizualizacji śladu
        trail_line.set_data(trail_x, trail_y)

        return rod_line, trail_line, bob1, bob2

    ani = animation.FuncAnimation(
        fig, update, frames=len(idx), init_func=init,
        blit=True, interval=1000 / playback_fps
    )
    plt.show()


if __name__ == '__main__':
    initial_conditions = prompt_initial_conditions()
    if initial_conditions is None:
        sys.exit("Anulowano symulację (zamknięto okno).")

    theta1_0, theta2_0, L1, L2, m1, m2 = initial_conditions
    x1, y1, x2, y2, idx, L1, L2, fps = run_simulation(
        theta1_0, theta2_0, L1, L2, m1, m2
    )
    animate(x1, y1, x2, y2, idx, L1, L2, fps)

#!/usr/bin/env python3
import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

class DoublePendulumApp:
    def __init__(self):
        # fizyka
        self.L1, self.L2 = 1.0, 1.0
        self.m1, self.m2 = 1.0, 1.0
        self.g = 9.81

        # parametry symulacji
        self.dt = 0.01
        self.tmax = 45.0
        self.t = np.arange(0, self.tmax + self.dt, self.dt)

        # parametry odtwarzania
        self.playback_fps = 60
        self.n_frames = int(self.tmax * self.playback_fps)
        self.idx = None  # wypełni się później

        # wartości startowe
        self.theta1_0 = 3 * np.pi / 7
        self.theta2_0 = 3 * np.pi / 4

        # kontenery na wynik
        self.x1 = self.y1 = self.x2 = self.y2 = None

    def equations(self, y, t):
        θ1, z1, θ2, z2 = y
        c, s = np.cos(θ1 - θ2), np.sin(θ1 - θ2)
        dz1 = (self.m2 * self.g * np.sin(θ2) * c
               - self.m2 * s * (self.L1 * z1 * z1 * c + self.L2 * z2 * z2)
               - (self.m1 + self.m2) * self.g * np.sin(θ1)) \
              / (self.L1 * (self.m1 + self.m2 * s * s))
        dz2 = ((self.m1 + self.m2) * (self.L1 * z1 * z1 * s
                                      - self.g * np.sin(θ2) + self.g * np.sin(θ1) * c)
               + self.m2 * self.L2 * z2 * z2 * s * c) \
              / (self.L2 * (self.m1 + self.m2 * s * s))
        return [z1, dz1, z2, dz2]

    def total_energy(self, Y):
        θ1, θ1d, θ2, θ2d = Y.T
        V = -(self.m1 + self.m2) * self.L1 * self.g * np.cos(θ1) \
            - self.m2 * self.L2 * self.g * np.cos(θ2)
        T = 0.5 * self.m1 * (self.L1 * θ1d) ** 2 \
            + 0.5 * self.m2 * ((self.L1 * θ1d) ** 2 + (self.L2 * θ2d) ** 2
                               + 2 * self.L1 * self.L2 * θ1d * θ2d * np.cos(θ1 - θ2))
        return T + V

    def prompt_initial_conditions(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        plt.subplots_adjust(left=0.1, bottom=0.4)
        ax.set_axis_off()
        s1 = Slider(plt.axes([0.1, 0.25, 0.8, 0.03]),
                    'θ₁ [rad]', 0, 2 * np.pi, valinit=self.theta1_0)
        s2 = Slider(plt.axes([0.1, 0.15, 0.8, 0.03]),
                    'θ₂ [rad]', 0, 2 * np.pi, valinit=self.theta2_0)
        btn = Button(plt.axes([0.4, 0.05, 0.2, 0.075]), 'Start')

        def on_start(evt):
            self.theta1_0, self.theta2_0 = s1.val, s2.val
            plt.close(fig)

        btn.on_clicked(on_start)
        plt.show()

    def run_simulation(self):
        # integracja
        y0 = [self.theta1_0, 0.0, self.theta2_0, 0.0]
        Y = odeint(self.equations, y0, self.t)
        # kontrola energii
        E0 = self.total_energy(np.array(y0).reshape(1, 4))
        if np.max(np.abs(self.total_energy(Y) - E0)) > 0.05:
            sys.exit("Drift energii za duży")
        # kartezjany
        θ1, θ2 = Y[:, 0], Y[:, 2]
        self.x1 = self.L1 * np.sin(θ1)
        self.y1 = -self.L1 * np.cos(θ1)
        self.x2 = self.x1 + self.L2 * np.sin(θ2)
        self.y2 = self.y1 - self.L2 * np.cos(θ2)

        # indeksy do 60 FPS
        self.idx = np.linspace(0, len(self.t) - 1, self.n_frames).astype(int)

    def animate(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-self.L1 - self.L2 - 0.1, self.L1 + self.L2 + 0.1)
        ax.set_ylim(-self.L1 - self.L2 - 0.1, self.L1 + self.L2 + 0.1)
        ax.set_aspect('equal')
        plt.axis('off')

        rod_line, = ax.plot([], [], lw=2, c='k')
        trail_line, = ax.plot([], [], lw=1, alpha=0.5)
        bob1 = Circle((0, 0), 0.05, fc='b', zorder=10)
        bob2 = Circle((0, 0), 0.05, fc='r', zorder=10)
        ax.add_patch(bob1)
        ax.add_patch(bob2)

        def init():
            rod_line.set_data([], [])
            trail_line.set_data([], [])
            bob1.center = (0, 0)
            bob2.center = (0, 0)
            return rod_line, trail_line, bob1, bob2

        def update(frame):
            i = self.idx[frame]
            rod_line.set_data([0, self.x1[i], self.x2[i]],
                              [0, self.y1[i], self.y2[i]])
            bob1.center = (self.x1[i], self.y1[i])
            bob2.center = (self.x2[i], self.y2[i])
            tr0 = max(0, i - int(1.0 / self.dt))
            trail_line.set_data(self.x2[tr0:i], self.y2[tr0:i])
            return rod_line, trail_line, bob1, bob2

        ani = animation.FuncAnimation(
            fig, update, frames=self.n_frames,
            init_func=init, blit=True,
            interval=1000 / self.playback_fps
        )
        plt.show()

    def run(self):
        self.prompt_initial_conditions()
        self.run_simulation()
        self.animate()


if __name__ == '__main__':
    DoublePendulumApp().run()

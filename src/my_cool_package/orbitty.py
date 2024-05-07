from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

FloatingPoint = np.dtype[np.floating[Any]]


class System:
    """
    Represents a system of gravitating particles.

    There are two constructors:

        System(masses, positions, momenta)

    takes explicit masses, positions, and momenta, while

        System.random(num_particles, num_dimensions, mass_mean, mass_width, x_width, p_width)

    takes parameters for random distributions and randomly generates particles.

    The gravitational scaling depends on the number of dimensions: there must be
    at least 2 dimensions, and a 2-dimensional system obeys a 1/r law, a 3-dimensional
    system obeys a 1/r^2 (inverse square) law, etc.

    The ``G`` attribute is the gravitational constant (scales the masses) and
    ``min_distance`` prevents gravitational forces from getting too large by
    truncating small distances to a specified minimum. (Particles are not
    infinitesimal points.)

    The ``step`` and ``steps`` methods run the simulation, changing the System's
    internal state, and ``plot`` returns the result as a Matplotlib animation.

    If Matplotlib or IPython is not installed, ``plot`` will raise a ModuleNotFoundError.
    """

    G: float = 3
    min_distance: float = 0.1

    # mass (m) is an array of some floating point type
    m: np.ndarray[tuple[int], FloatingPoint]

    # position (x) and momentum (p) have 2 components (num_particles, num_dimensions)
    x: np.ndarray[tuple[int, int], FloatingPoint]
    p: np.ndarray[tuple[int, int], FloatingPoint]

    @classmethod
    def random(
        cls,
        num_particles: int,
        num_dimensions: int,
        mass_mean: float,
        mass_width: float,
        x_width: float,
        p_width: float,
        rng: None | np.random.Generator = None,
    ) -> System:
        """
        Generate a system of gravitating particles randomly.

        Args:
            num_particles (int): The number of particles to generate.
            num_dimensions (int): The number of dimensions in each particle's
                positions and momenta.
            mass_mean (float): The mean value of particle mass, generated with
                a gamma distribution.
            mass_width (float): The "theta" parameter for the mass's gamma
                distribution. The variance of the masses is theta times the mean.
            x_width (float): The "sigma" (standard deviation) of positions,
                generated as a normal (Gaussian) distribution with zero mean.
            p_width (float): The "sigma" (standard deviation) of momenta,
                generated as a normal (Gaussian) distribution with zero mean.
            rng (None or np.random.Generator): A random number generator from NumPy.
                If None, a ``np.random.default_rng`` is generated instead.
        """

        if rng is None:
            rng = np.random.default_rng()

        m = rng.gamma(mass_mean / mass_width, mass_width, num_particles)
        x = rng.normal(0, x_width, (num_particles, num_dimensions))
        p = rng.normal(0, p_width, (num_particles, num_dimensions))
        return cls(m, x, p)

    def __init__(self, m: npt.ArrayLike, x: npt.ArrayLike, p: npt.ArrayLike):
        """
        Initialize a system of gravitating particles with explicit values.

        Args:
            m (npt.ArrayLike): Ordered collection of masses (1-dimensional).
            x (npt.ArrayLike): Collection of positions in the same order.
                The first axis has one subarry per particle; the second axis
                has one position in each dimension.
            p (npt.ArrayLike): Collection of momenta in the same order.
                The first axis has one subarry per particle; the second axis
                has one momentum in each dimension.
        """

        self.x, self.p = np.broadcast_arrays(x, p)
        assert self.x.shape == self.p.shape
        if len(self.x.shape) != 2:
            err = f"arrays of position and momentum must each have 2 components, not {len(self.x.shape)}"  # type: ignore[unreachable]
            raise ValueError(err)
        if self.num_dimensions < 2:
            err = "number of dimensions must be at least 1"
            raise ValueError(err)

        self.m, _ = np.broadcast_arrays(m, self.x[:, 0])
        assert len(self.m) == len(self.x)
        if len(self.m.shape) != 1:
            err = f"array of masses must have only 1 component, not {len(self.m.shape)}"
            raise ValueError(err)

        if issubclass(self.m.dtype.type, np.integer):  # type: ignore[unreachable]
            self.m = self.m.astype(np.float64)
        if issubclass(self.x.dtype.type, np.integer):
            self.x = self.x.astype(np.float64)
        if issubclass(self.p.dtype.type, np.integer):
            self.p = self.p.astype(np.float64)
        if not issubclass(self.m.dtype.type, np.floating):
            err = f"masses must have floating-point type, not {self.m.dtype}"
            raise TypeError(err)
        if not issubclass(self.x.dtype.type, np.floating):
            err = f"positions must have floating-point type, not {self.m.dtype}"
            raise TypeError(err)
        if not issubclass(self.p.dtype.type, np.floating):
            err = f"momenta must have floating-point type, not {self.m.dtype}"
            raise TypeError(err)

        self.history: list[System.Step] = [self.Step(0, self.x, self.p)]

    class Step:
        """
        Represents one time-step in the System's ``history``.
        """

        def __init__(
            self,
            t: float,
            x: np.ndarray[tuple[int, int], FloatingPoint],
            p: np.ndarray[tuple[int, int], FloatingPoint],
        ):
            self.t, self.x, self.p = t, x, p

        def __repr__(self) -> str:
            return f"<Step t={self.t} x={self.x.tolist()} p={self.p.tolist()}>"

    def __repr__(self) -> str:
        return f"<System of {self.num_particles} particles in {self.num_dimensions} dimensions>"

    @property
    def num_particles(self) -> int:
        """
        The number of particles in the System.
        """

        return self.x.shape[0]

    @property
    def num_dimensions(self) -> int:
        """
        The number of dimensions in each position and momentum.
        """

        return self.x.shape[1]  # type: ignore[no-any-return]

    @property
    def forces(self) -> np.ndarray[tuple[int, int], FloatingPoint]:
        """
        The total force, as a vector on each particle, due to gravitational
        attraction to all other particles in the System.

        This array has the same shape as ``x`` and ``p``.
        """

        # indexes to pick out (particle 1, particle 2) pairs, for all pairs
        p1, p2 = np.triu_indices(len(self.x), k=1)
        # pairwise (pw) displacements between all particle pairs
        pw_displacement = self.x[p2] - self.x[p1]
        # pairwise displacement is a sum in quadrature over all dimensions
        pw_distance = np.maximum(
            np.sqrt(np.sum(pw_displacement**2, axis=-1)), self.min_distance
        )
        # direction is a unit vector
        pw_direction = pw_displacement / pw_distance[:, np.newaxis]
        m1 = self.m[p1, np.newaxis]
        m2 = self.m[p2, np.newaxis]
        # 1/r in 2D, 1/r**2 in 3D, 1/r**3 in 4D...
        power = self.num_dimensions - 1
        # law of universal gravitation
        pw_force = self.G * m1 * m2 * pw_direction / pw_distance[:, np.newaxis] ** power
        # vector sum over pairs for each particle, np.add.at inverts p1, p2 indexing
        total_force = np.zeros_like(self.x)
        np.add.at(total_force, p1, pw_force)
        np.add.at(total_force, p2, -pw_force)
        return total_force

    def step(self, dt: float = 0.1) -> None:
        """
        Simulate the System for one time-step.

        Args:
            dt (float): Time interval to simulate. The smaller this value is,
                the more precise the simulation will be.

        Uses a kick-drift-kick method to control numerical error. Consequently,
        this function calls ``forces`` twice.

        This method changes the state of the System, including its ``history``.
        """

        half_dt = dt / 2
        # half kick: update p by a half time-step using current positions
        self.p = self.p + self.forces * half_dt
        # full drift: update x by a full time-step using new momenta
        self.x = self.x + self.p * dt / self.m[:, np.newaxis]
        # half kick: update p by another half time-step using new positions
        self.p = self.p + self.forces * half_dt
        # save the history
        self.history.append(self.Step(self.history[-1].t + dt, self.x, self.p))

    def steps(self, n: int, dt: float = 0.01) -> None:
        """
        Simulate the System for ``n`` time-steps.

        Args:
            n (int): Number of time-steps.
            dt (float): Time interval to simulate. The smaller this value is,
                the more precise the simulation will be.

        This method changes the state of the System, including its ``history``.
        """

        for _ in range(n):
            self.step(dt=dt)

    @property
    def t_history(self):
        """
        Get the history of time-steps as an array.

        The 1 axis is

        * time-steps
        """

        return np.array([step.t for step in self.history])

    @property
    def x_history(self):
        """
        Get the history of x positions as an array.

        The 3 axes are

        * time-steps
        * particles
        * dimensions
        """

        x = np.empty((len(self.history), self.num_particles, 2))
        for i, step in enumerate(self.history):
            for j in range(self.num_particles):
                x[i, j, :] = step.x[j, :2]
        return x

    @property
    def p_history(self):
        """
        Get the history of p momenta as an array.

        The 3 axes are

        * time-steps
        * particles
        * dimensions
        """

        p = np.empty((len(self.history), self.num_particles, 2))
        for i, step in enumerate(self.history):
            for j in range(self.num_particles):
                p[i, j, :] = step.p[j, :2]
        return p

    def plot(
        self,
        figsize: tuple[int, int] = (5, 5),
        method: Literal["to_jshtml", "to_html5_video"] = "to_jshtml",
        num_frames: int = 100,
        frame_ms: int = 50,
    ) -> Any:
        """
        Present the time-evolution of the System as a Matplotlib animation.

        Be sure to call ``step`` or ``steps`` before this function, so that there
        is something to plot!

        Args:
            figsize (tuple[int, int]): Matplotlib figure size.
            method ("to_jshtml", "to_html5_video"): Name of the animation-to-HTML
                method. ``to_jshtml`` always works, and ``to_html5_video`` works
                if video codecs are available (JupyterLab but not JupyterLite).
            num_frames (int): Number of frames to render in the animation, which
                can be fewer than the number of simulated time-steps.
            frame_ms (int): Number of milliseconds between each frame.
        """

        import matplotlib.pyplot as plt  # pylint: disable=E0401, C0415
        from IPython.display import HTML  # pylint: disable=E0401, C0415
        from matplotlib import animation  # pylint: disable=E0401, C0415

        fig, ax = plt.subplots(figsize=figsize)

        x = self.x_history

        x0 = np.mean(x[:, :, 0])
        y0 = np.mean(x[:, :, 1])
        scale = np.percentile(np.max(abs(x), axis=0), 75) * 1.5
        ax.set(xlim=(x0 - scale, x0 + scale), ylim=(y0 - scale, y0 + scale))

        if len(x) > num_frames:
            x = x[:: len(x) // num_frames]

        lines = []
        for j in range(self.num_particles):
            lines.append(ax.plot(x[:1, j, 0], x[:1, j, 1])[0])
        dots = ax.scatter(x[0, :, 0], x[0, :, 1], color="black")

        def update(i: int) -> list[Any]:
            for j, line in enumerate(lines):
                line.set_xdata(x[:i, j, 0])
                line.set_ydata(x[:i, j, 1])
            dots.set_offsets(x[i, :, :])
            return [*lines, dots]

        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=len(x), interval=frame_ms, blit=True
        )

        out = HTML(getattr(ani, method)())
        plt.close()
        return out

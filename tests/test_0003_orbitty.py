from __future__ import annotations

import pickle

import numpy as np
import pytest

from my_cool_package import orbitty


def test_values():
    system = orbitty.System(
        m=[1000, 1], x=[[0, 0, 0], [0, 10, 0]], p=[[10, 0, 0], [-10, 0, 0]]
    )
    system.steps(1000)

    with open("tests/samples/orbitty-samples.pkl", "rb") as file:
        t_history, x_history, p_history = pickle.load(file)

    np.testing.assert_allclose(system.t_history, t_history)
    np.testing.assert_allclose(system.x_history, x_history)
    np.testing.assert_allclose(system.p_history, p_history)


def center_of_mass(system: orbitty.System) -> list[float]:
    return np.sum(  # type: ignore[no-any-return]
        system.m[:, np.newaxis] * system.x / np.sum(system.m), axis=0
    ).tolist()


def test_center_of_mass():
    system = orbitty.System(
        m=[1000, 1], x=[[0, 0, 0], [0, 10, 0]], p=[[10, 0, 0], [-10, 0, 0]]
    )

    initial = center_of_mass(system)
    for _ in range(1000):
        system.step()
        assert center_of_mass(system) == pytest.approx(initial)


def total_momentum(system: orbitty.System) -> list[float]:
    return np.sum(system.p, axis=0).tolist()  # type: ignore[no-any-return]


def test_total_momentum():
    system = orbitty.System(
        m=[1000, 1, 1],
        x=[[0, 0, 0], [0, 10, 0], [0, 11, 0]],
        p=[[0, 0, 0], [-16, 0, 0], [-13, 0, 0]],
    )

    initial = total_momentum(system)
    for _ in range(1000):
        system.step()
        assert total_momentum(system) == pytest.approx(initial)


def total_energy(system: orbitty.System) -> float:
    # KE = 1/2 m |v|^2 and p = mv, so KE = 1/2 |p|^2 / m
    kinetic = np.sum(0.5 * np.sum(system.p**2, axis=1) / system.m)

    # gravitational force -> potential integration in 3D
    assert system.num_dimensions == 3
    # indexes to pick out (particle 1, particle 2) pairs, for all pairs
    p1, p2 = np.triu_indices(len(system.x), k=1)
    # pairwise (pw) displacements between all particle pairs
    pw_displacement = system.x[p2] - system.x[p1]
    # pairwise displacement is a sum in quadrature over all dimensions
    pw_distance = np.sqrt(np.sum(pw_displacement**2, axis=-1))
    # PE = -G m1 m2 / distance (for each pair)
    pw_potential = -system.G * system.m[p1] * system.m[p2] / pw_distance
    # sum over pairs to get the potential for each particle
    particle_potential = np.zeros_like(system.m)
    np.add.at(particle_potential, p1, pw_potential)
    np.add.at(particle_potential, p2, pw_potential)
    # avoid double-counting (particle 1, particle 2) and (particle 2, particle 1)
    potential = 0.5 * np.sum(particle_potential)

    return potential + kinetic  # type: ignore[no-any-return]


def test_total_energy():
    p1 = 0.347111
    p2 = 0.532728
    system = orbitty.System(
        m=[1, 1, 1],
        x=[[-1, 0, 0], [1, 0, 0], [0, 0, 0]],
        p=[[p1, p2, 0], [p1, p2, 0], [-2 * p1, -2 * p2, 0]],
    )
    system.G = 1

    initial = total_energy(system)
    for _ in range(10000):
        system.step(dt=0.001)  # smaller time-steps instead of loosening pytest.approx
        assert total_energy(system) == pytest.approx(initial)


def test_random():
    rng = np.random.default_rng(seed=12345)

    system = orbitty.System.random(
        num_particles=20,
        num_dimensions=3,
        mass_mean=10,
        mass_width=1,
        x_width=100,
        p_width=10,
        rng=rng,
    )

    initial_momentum = total_momentum(system)
    initial_energy = total_energy(system)
    for _ in range(10000):
        system.step(dt=0.01)
        assert total_momentum(system) == pytest.approx(initial_momentum)
        assert total_energy(system) == pytest.approx(initial_energy, rel=1e-5)

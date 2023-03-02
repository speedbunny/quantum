#Loop Quantum Gravity Experiment 1
#Sarah Eaglesfield
#March 2023

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set the size of the lattice
L = 10

# Set the number of spins in the lattice
N = L**3

# Set the number of possible spin values
spin_values = np.array([-1, 0, 1])

# Generate random initial spin configuration
spins = np.random.choice(spin_values, size=N)

# Define the Hamiltonian
def hamiltonian(spins):
    # Calculate the energy of the lattice
    energy = 0
    for i in range(N):
        for j in range(N):
            energy += np.exp(1j * np.pi * spins[i] * spins[j])
    return energy

# Define the Metropolis-Hastings algorithm
def metropolis_hastings(spins, beta):
    # Pick a random spin to flip
    i = np.random.randint(N)
    # Calculate the energy change
    energy_change = 2 * hamiltonian(spins) * np.exp(-2 * beta * np.pi * spins[i] * np.sum(spins))
    # If the energy change is negative or satisfies the Boltzmann distribution, flip the spin
    if energy_change < 0 or np.random.rand() < np.exp(-beta * energy_change):
        spins[i] = -spins[i]

# Define the simulation
def spin_foam_model(beta, steps):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for step in range(steps):
        # Run the Metropolis-Hastings algorithm for every spin
        for i in range(N):
            metropolis_hastings(spins, beta)
        # Visualize the lattice
        padded_spins = np.pad(spins.reshape((L, L, L)), [(1, 1), (1, 1), (1, 1)], mode='constant')
        x, y, z = np.indices((L+2, L+2, L+2))
        voxels = padded_spins.astype(float)
        ax.voxels(x, y, z, voxels, edgecolor='k')
        plt.title(f'Step {step}')
        plt.show()

# Run the simulation
spin_foam_model(0.1, 10)

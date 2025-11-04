import numpy as np


def calculate_energies(pos, vel, mass, bodies, G, softening):
    arr = []
    for i in range(int(len(pos) / bodies)):
        arr.append(
            _calculate_energies(
                pos[(i * bodies) : (i * bodies + bodies)],
                vel[(i * bodies) : (i * bodies + bodies)],
                mass[(i * bodies) : (i * bodies + bodies)],
                G,
                softening,
            )
        )
    return np.array(arr)


def _calculate_energies(pos, vel, mass, G, softening):
    # Kinetic Energy:
    KE = 0.5 * np.sum(np.sum(mass * vel**2))

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2 + softening**2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    # sum over upper triangle, to count each interaction only once
    PE = G * np.sum(np.sum(np.triu(-(mass * mass.T) * inv_r, 1)))

    return KE, PE

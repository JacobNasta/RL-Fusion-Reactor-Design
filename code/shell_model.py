import numpy as np
from scipy.integrate import quad


class ShellModel: # a simplified model used to check if a nucleus is bound and if so, evaluate the binding energy 

    orbital_levels = [
        "1s_1/2",
        "1p_3/2",
        "1p_1/2",
        "1d_5/2"
    ]

    proton_energy = 938.27208816
    neutron_energy = 939.56542052

    def orbitinfo(self, level, A, well_depth=20, spin_orbit_strength=-10):
        orbital_letters = "spdfgh"
        n_and_l, j = level.split("_")
        n = int(n_and_l[0])
        orbital_letter = n_and_l[1]
        l = orbital_letters.index(orbital_letter)
        j_top, j_bottom = j.split("/")
        j = int(j_top) / int(j_bottom)
        capacity = int(2 * j + 1)
        radial_n = n - 1
        spin = 0.5
        spin_orbit = (
                j * (j + 1) - l * (l + 1) - spin * (spin + 1)
        ) / 2
        hbar_omega = 14.55 / A**(1/3)
        energy = well_depth - hbar_omega * (2 * radial_n + l + 1.5) - spin_orbit_strength * spin_orbit
        return {
            "level": level,
            "n": n,
            "l": l,
            "j": j,
            "capacity": capacity,
            "energy": energy
        }

    def fillorbitals(self, Z, N):
        orbitalscapacityZ = []
        orbitalscapacityN = []
        A = Z + N
        for level in self.orbital_levels:
            capacity = self.orbitinfo(level, A)["capacity"]
            if Z >= capacity:
                orbitalscapacityZ.append(capacity)
                Z -= capacity
            else:
                orbitalscapacityZ.append(Z)
                Z = 0
        for level in self.orbital_levels:
            capacity = self.orbitinfo(level, A)["capacity"]
            if N >= capacity:
                orbitalscapacityN.append(capacity)
                N -= capacity
            else:
                orbitalscapacityN.append(N)
                N = 0
        return orbitalscapacityZ, orbitalscapacityN

    def energycalc(self, Z, N, orbitalscapacityZ, orbitalscapacityN):
        A = Z + N
        energy = 0
        for i in range(len(orbitalscapacityZ)):
            energy += self.orbitinfo(self.orbital_levels[i], A)["energy"] * (orbitalscapacityZ[i] + orbitalscapacityN[i])
        return energy

    def interactionenergy(self, Z, N):
        A = Z + N
        if A <= 1:
            return 0
        pairing_strength = 11.52 / A**0.5
        coulomb_strength = 1.96
        asymmetry_strength = 18.19
        if A < 4:
            pairing = 0
        elif Z % 2 == 0 and N % 2 == 0:
            pairing = pairing_strength
        elif Z % 2 == 1 and N % 2 == 1:
            pairing = -pairing_strength
        else:
            pairing = 0
        coulomb = -coulomb_strength * Z * (Z - 1) / A**(1/3)
        asymmetry = -asymmetry_strength * (N - Z)**2 / A
        return pairing + coulomb + asymmetry

    def totalenergy(self, Z, N, orbitalscapacityZ, orbitalscapacityN):
        free_energy = (Z * self.proton_energy + N * self.neutron_energy)
        binding_energy = self.energycalc(Z, N, orbitalscapacityZ, orbitalscapacityN) + self.interactionenergy(Z, N)
        binding_energy = max(0, binding_energy)
        return free_energy - binding_energy

    def bindingenergy(self, Z, N):
        orbitalsZ, orbitalsN = self.fillorbitals(Z, N)
        if Z + N <= 1:
            return 0
        free_energy = Z * self.proton_energy + N * self.neutron_energy
        return free_energy - self.totalenergy(Z, N, orbitalsZ, orbitalsN)

    from functools import lru_cache
    @lru_cache(maxsize=None)
    def bound_nucleus(self, Z, N):
        combinations = []
        for Z1 in range(Z + 1):
            for Z2 in range(Z - Z1 + 1):
                Z3 = Z - Z1 - Z2
                for N1 in range(N + 1):
                    for N2 in range(N - N1 + 1):
                        N3 = N - N1 - N2
                        nuclei = [(Z1, N1), (Z2, N2), (Z3, N3)]
                        total_binding = 0
                        for nucleus_Z, nucleus_N in nuclei:
                            if nucleus_Z + nucleus_N == 0:
                                continue
                            total_binding += self.bindingenergy(nucleus_Z, nucleus_N)
                        combinations.append((total_binding, nuclei))
        return max(combinations, key=lambda x: x[0])


class resonant_model: # computes the cross section and reactivity at a given temperature 

    e = 1.602176634e-19
    hbar = 1.054571817e-34
    epsilon_0 = 8.8541878128e-12
    k = 1.380649e-23
    u = 1.66053906660e-27
    mev_to_joules = 1.602176634e-13

    def __init__(self):
        self.shell = ShellModel()

    def nuclear_radii(self, particle):
        A = particle.Z + particle.N
        return 1.2e-15 * A**(1/3)

    def reduced_mass(self, fuel1, fuel2):
        mass1 = (fuel1.Z + fuel1.N) * self.u
        mass2 = (fuel2.Z + fuel2.N) * self.u
        return mass1 * mass2 / (mass1 + mass2)

    def total_angular_momentum(self, particle):
        orbitalcapacityZ, orbitalcapacityN = self.shell.fillorbitals(particle.Z, particle.N)
        Zshell_j = 0
        Nshell_j = 0
        for i in range(len(orbitalcapacityZ)):
            if orbitalcapacityZ[i] % 2 == 1:
                Zshell_j = self.shell.orbitinfo(self.shell.orbital_levels[i], particle.Z + particle.N)["j"]
        for i in range(len(orbitalcapacityN)):
            if orbitalcapacityN[i] % 2 == 1:
                Nshell_j = self.shell.orbitinfo(self.shell.orbital_levels[i], particle.Z + particle.N)["j"]
        return Zshell_j + Nshell_j

    def energy_release(self, fuel1, fuel2):
        reactant_binding = self.shell.bindingenergy(fuel1.Z, fuel1.N) + self.shell.bindingenergy(fuel2.Z, fuel2.N)
        product_binding = self.shell.bound_nucleus(fuel1.Z + fuel2.Z, fuel1.N + fuel2.N)[0]
        return max(0, product_binding - reactant_binding)

    def S_factor(self, fuel1, fuel2, E):
        radius1 = self.nuclear_radii(fuel1)
        radius2 = self.nuclear_radii(fuel2)
        nuclear_area = np.pi * (radius1 + radius2)**2
        energy_release = self.energy_release(fuel1, fuel2)
        nuclear_energy = (1 + energy_release) * self.mev_to_joules
        angular_momentum = self.total_angular_momentum(fuel1) + self.total_angular_momentum(fuel2)
        angular_factor = 1 / (1 + angular_momentum)
        energy_factor = 1 + E / nuclear_energy
        return nuclear_area * nuclear_energy * angular_factor * energy_factor

    def cross_section(self, fuel1, fuel2, E):
        reduced_mass = self.reduced_mass(fuel1, fuel2)
        v = (2 * E / reduced_mass) ** 0.5
        eta = fuel1.Z * fuel2.Z * self.e ** 2 / (4 * np.pi * self.epsilon_0 * self.hbar * v)
        S = self.S_factor(fuel1, fuel2, E)
        return S / E * np.exp(-2 * np.pi * eta)

    def reactivity(self, fuel1, fuel2, T):
        reduced_mass = self.reduced_mass(fuel1, fuel2)
        thermal_energy = self.k * T
        prefactor = (8 / (np.pi * reduced_mass)) ** 0.5 / thermal_energy ** 1.5
        def integrand(E):
            return self.cross_section(fuel1, fuel2, E) * E * np.exp(-E / thermal_energy)
        integral = quad(integrand, 1e-25, 50 * thermal_energy)[0]
        return prefactor * integral

    def non_resonant_model(self, fuel1, fuel2, T):
        reduced_mass = self.reduced_mass(fuel1, fuel2)
        E = 3/2 * self.k * T
        v = (2 * E / reduced_mass)**0.5
        eta = fuel1.Z * fuel2.Z * self.e**2 / (4 * np.pi * self.epsilon_0 * self.hbar * v)
        S = self.S_factor(fuel1, fuel2, E)
        cross_section = S / E * np.exp(-2 * np.pi * eta)
        return cross_section

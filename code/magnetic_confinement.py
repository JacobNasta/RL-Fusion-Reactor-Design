import numpy as np


class Reactor_geometry: # evaluates the geometry of the plasma being stored by the field

    def volume(self, major_radius, minor_radius, elongation, triangularity, squareness):
        shape_factor = (1 + 0.25 * squareness - 0.1 * triangularity ** 2)
        if major_radius == 0:
            volume = 4 / 3 * np.pi * minor_radius **3 * elongation * shape_factor
        else:
            volume = 2 * major_radius * np.pi * minor_radius**2 * np.pi * elongation * shape_factor
        return volume

    def surface_area(self, major_radius, minor_radius, elongation, triangularity, squareness):
        shape_factor = (1 + 0.25 * squareness - 0.1 * triangularity ** 2)
        if major_radius == 0:
            surface_area = 4 * np.pi * minor_radius**2 * elongation * shape_factor
        else:
            surface_area = 2 * np.pi * minor_radius * 2 * np.pi * major_radius * elongation * shape_factor
        return surface_area

    def cross_section(self, minor_radius, elongation, triangularity, squareness):
        shape_factor = (1 + 0.25 * squareness - 0.1 * triangularity ** 2)
        cross_section = np.pi * minor_radius**2 * elongation * shape_factor
        return cross_section

    def aspect_ratio(self, major_radius, minor_radius):
        if major_radius == 0:
            return 0
        return major_radius / minor_radius

    def inverse_aspect_ratio(self, major_radius, minor_radius):
        if major_radius == 0:
            return 0
        return minor_radius / major_radius

    def volume_to_surface(self, major_radius, minor_radius, elongation, triangularity, squareness):
        volume = self.volume(major_radius, minor_radius, elongation, triangularity, squareness)
        surface_area = self.surface_area(major_radius, minor_radius, elongation, triangularity, squareness)
        return volume / surface_area

    def magnetic_path_length(self, major_radius, minor_radius):
        if major_radius == 0:
            return 2 * np.pi * minor_radius
        return 2 * np.pi * major_radius

class Magnetic_field: # evaluates the strength and energy stored/required to produce the magnetic field

    def magentic_field_strength(self, N, current, major_radius, minor_radius, permeability=4 * np.pi * 10 ** -7):
        if major_radius == 0:
            magnetic_path_length = 2 * np.pi * minor_radius
        else:
            magnetic_path_length = 2 * np.pi * major_radius
        return permeability * N * current / magnetic_path_length

    def magnetic_energy(self, B, V, permeability = 4*np.pi*10**-7):
        return B**2 * V / (2 * permeability)


class plasma: # evaluates certain parameters used to define the success of the model

    def beta(self, B, n, T, mu = 4*np.pi*10**-7, k = 1.380649e-23):
        plasma_pressure = 2 * n * k * T
        magnetic_pressure = B**2 / (2 * mu)
        return plasma_pressure/magnetic_pressure

    def thermal_energy(self, T, V, n, k = 1.380649e-23):
        return 3 * n * k * T * V

    def energy_loss(self, Z, n, T, B, V, surface_area, brem_scaling, synch_scaling, surface_scaling):
        bremsstrahlung_energy = n ** 2 * Z ** 2 * T ** (1 / 2) * V * brem_scaling
        synchrotron = n * T ** 2 * B ** 2 * V * synch_scaling
        surface_loss = surface_area * surface_scaling
        return bremsstrahlung_energy + synchrotron + surface_loss

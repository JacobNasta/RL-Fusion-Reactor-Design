from shell_model import ShellModel
from shell_model import resonant_model
from magnetic_confinement import Reactor_geometry
from magnetic_confinement import Magnetic_field
from magnetic_confinement import plasma
import numpy as np


class power: # combines the physics built in the other classes to define the net power output of the reactor

    mev_to_joules = 1.602176634e-13

    def __init__(self):
        self.shell = ShellModel()
        self.resonant = resonant_model()
        self.geometry = Reactor_geometry()
        self.magnetic = Magnetic_field()
        self.plasma = plasma()

    def energy_per_reaction(self, fuel1, fuel2):
        BE_before = self.shell.bindingenergy(fuel1.Z, fuel1.N) + self.shell.bindingenergy(fuel2.Z, fuel2.N)
        products = self.shell.bound_nucleus(fuel1.Z + fuel2.Z, fuel1.N + fuel2.N)
        BE_after = products[0]
        energy_release = BE_after - BE_before
        return max(0, energy_release)

    def reaction_rate(self, fuel1, fuel2, n, V, T):
        n1 = n / 2
        n2 = n / 2
        reactivity = self.resonant.reactivity(fuel1, fuel2, T)
        if fuel1.Z == fuel2.Z and fuel1.N == fuel2.N:
            reaction_rate = 0.5 * n1 * n2 * reactivity * V  # identical particles: avoid double-counting pairs
        else:
            reaction_rate = n1 * n2 * reactivity * V
        return reaction_rate

    def power_output(self, fuel1, fuel2, n, V, T):
        energy_release = self.energy_per_reaction(fuel1, fuel2)
        reaction_rate = self.reaction_rate(fuel1, fuel2, n, V, T)
        poweroutput = energy_release * self.mev_to_joules * reaction_rate
        return poweroutput

    def power_loss(self, Z, n, V, T, B, surface_area, brem_scaling, synch_scaling, surface_scaling):
        plasma_loss = self.plasma.energy_loss(Z, n, T, B, V, surface_area, brem_scaling, synch_scaling, surface_scaling)
        return plasma_loss

    def total_power(self, fuel1, fuel2, n, V, T, B, surface_area, brem_scaling, synch_scaling, surface_scaling):
        power_output = self.power_output(fuel1, fuel2, n, V, T)
        power_loss = self.power_loss(fuel1.Z + fuel2.Z, n, V, T, B, surface_area, brem_scaling, synch_scaling, surface_scaling)
        return power_output - power_loss


class stability: # as the name suggests, combines the physics built to define parameters which will determine the models success (related to the stability) 

    def __init__(self):
        self.plasma = plasma()

    def confinement_time(self, Z, n, T, B, V, surface_area, brem_scaling, synch_scaling, surface_scaling):
        thermal_energy = self.plasma.thermal_energy(T, V, n)
        power_loss = self.plasma.energy_loss(Z, n, T, B, V, surface_area, brem_scaling, synch_scaling, surface_scaling)
        if power_loss <= 0:
            return 0
        return thermal_energy / power_loss

    def beta(self, B, n, T):
        return self.plasma.beta(B, n, T)


class reactor: # wraps everything to provide all the final metrics used to judge the code on 

    def __init__(self):
        self.geometry = Reactor_geometry()
        self.magnetic = Magnetic_field()
        self.plasma = plasma()
        self.power = power()
        self.stability = stability()

    def reactor_performance(self, fuel1, fuel2, major_radius, minor_radius, elongation, triangularity, squareness,
                            N, current, n, T, brem_scaling, synch_scaling, surface_scaling):

        V = self.geometry.volume(major_radius, minor_radius, elongation, triangularity, squareness)
        surface_area = self.geometry.surface_area(major_radius, minor_radius, elongation, triangularity, squareness)
        cross_section = self.geometry.cross_section(minor_radius, elongation, triangularity, squareness)
        volume_to_surface = self.geometry.volume_to_surface(major_radius, minor_radius, elongation, triangularity, squareness)
        B = self.magnetic.magentic_field_strength(N, current, major_radius, minor_radius)
        magnetic_energy = self.magnetic.magnetic_energy(B, V)
        beta = self.stability.beta(B, n, T)
        confinement_time = self.stability.confinement_time(fuel1.Z + fuel2.Z, n, T, B, V, surface_area, brem_scaling, synch_scaling, surface_scaling)
        total_power = self.power.total_power(fuel1, fuel2, n, V, T, B, surface_area, brem_scaling, synch_scaling, surface_scaling)
        return {
            "volume": V,
            "surface_area": surface_area,
            "cross_section": cross_section,
            "volume_to_surface": volume_to_surface,
            "magnetic_field": B,
            "magnetic_energy": magnetic_energy,
            "beta": beta,
            "confinement_time": confinement_time,
            "total_power": total_power
        }

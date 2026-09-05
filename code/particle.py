class Particle: # holds the proton and neutron number for the particle and calculates the mass number

    def __init__(self, Z, N):
        self.Z = Z
        self.N = N
        self.A = Z + N

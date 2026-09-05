# Report

## 1. Introduction

This project set out to enable an agent to design a fusion reactor from a self-built physics environment. The goal was to maximise energy output by maximising confinement time and power output whilst minimising the power lost via running the reactor. The end result was that the agent converged to a small, near-spherical geometry, as it could not find a set up that produced a net positive output and thought to instead minimise the losses. However, several parameters matched up with accepted values. The main focus of this report will be why some parameters emerged successfully whereas others didn't. Note that several scaling constants in the underlying physics (bremsstrahlung, synchrotron, and surface loss scaling, and the shape-factor coefficients) are placeholders rather than values derived from first principles, chosen to scale the above contributors realistically.

## 2. Issue 1 — The Shell Model

In this project, the shell model is responsible for two things: calculating the binding energy of a nucleus (directly contributing to the Q-value of any reaction), and for reaction products. It finds the most bound way to arrange the resulting nucleons, comparing the combined binding energy against every way of splitting the same proton and neutron numbers into up to three fragments. This checks whether staying together is actually the lowest-energy outcome.

That second check only ever runs on the fusion product, not on the fuel itself. There is nothing in this model that tests whether an input fuel nucleus is realistic or stable before the agent is allowed to pick it. This is the direct cause of the biggest issue in the final result: the agent settled on deuterium and hydrogen-4 (one proton, three neutrons) as fuel. It correctly learned that lower proton numbers reduce the Coulomb barrier, but hydrogen-4 doesn't exist as a stable nucleus in reality, and its appearance here is evidence the shell model did not capture the real nuclear structure closely enough. Furthermore, fuel stability was never something the reward could check for directly. A fix for this would be testing whether either neighbouring isotope is more bound, similar to how the product method already works, and walking toward it if so. In practice, this doesn't just remove the unstable nuclei, but means the agent discovers the single peak in binding energy for each proton number. The result is some genuinely stable nuclei get removed in order for the unstable nuclei to not be selected.

In reality, this is a much harder problem to solve. Real shell structure shifts with mass number and deformation, and binding energies at low mass numbers are usually taken from measurement rather than derived. The toy shell model was used deliberately, to avoid handing the agent known facts and instead let it work from the underlying physics alone. Building an accurate shell model is well beyond the scope of this project, so this was documented as a limitation rather than fixed.

## 3. Issue 2 — The Geometry

The agent's best result was a tiny sphere, not a torus, and not an attempt to maximise volume. The usual argument for building larger reactors is that power output scales with volume while surface losses scale with surface area, so volume grows faster than surface area as the reactor gets bigger. That logic was not evident here. Because net power output never went positive, the agent's goal of maximising power effectively became minimising loss. The most effective way to do that ended up being to make the reactor as small as possible. This isn't the agent discovering that a sphere beats a tokamak, it's a consequence of the underlying reaction physics never being strong enough to let it reach a positive power output in the first place.

## 4. Parameter-by-Parameter Comparison

| Parameter | Agent's best result | Generally accepted (D-T tokamak) |
|---|---|---|
| Fuel | H-4 (Z=1, N=3) + D (Z=1, N=1) | D + T (Z=1, N=1 and Z=1, N=2) |
| Major radius | 0 m (no torus) | ~1.5–6 m |
| Minor radius | 0.05 m | ~0.5–2 m |
| Elongation | 1.914 | ~1.7–2.0 |
| Triangularity | −0.127 | ~+0.3–0.5 |
| Squareness | −0.006 | ~0 |
| Magnetic field | 5.96 T | ~5–13 T |
| Density | 3.52×10²⁰ m⁻³ | ~1×10²⁰ m⁻³ |
| Temperature | 6.34×10⁷ K | ~1–2×10⁸ K |
| Volume | 0.001 m³ | order 100s of m³ |
| Beta | 0.0436 | ~0.02–0.05 |
| Confinement time | 2.03×10⁻² s | ~3–6 s (target) |
| Total power | −4.53×10⁴ W | net positive (target) |

### Fuel selection
The agent correctly learned that lower proton numbers reduce the Coulomb barrier between the two nuclei, making fusion easier to reach. It failed to account for hydrogen-4's real-world instability, since nothing in the model checks fuel stability directly.

### Major radius and minor radius
As above — because the agent could never find a configuration with positive power output, shrinking to a small sphere was the best available strategy for minimising loss.

### Elongation
This is likely a coincidence rather than a genuine result. Elongation enters both the volume and surface-area formulas as the same multiplicative factor, so it cancels out of the volume-to-surface ratio entirely. There's no rewards pushing it toward any particular value, realistic or otherwise. This is worth revisiting in future versions of the model, but the major issue in geometry is the small sphere choice, not the elongation. 

### Triangularity
Triangularity only enters the geometry through the shape-factor equation as a squared term, so it penalises volume equally regardless of sign. The model has no way to distinguish +0.127 from −0.127, meaning the negative sign the agent landed on carries no real value. The magnitude does line up with the agent's general shrink-to-minimise-loss strategy, since any nonzero triangularity slightly reduces volume.

### Squareness
Squareness enters volume and surface area through the same shape-factor equation as elongation, so like elongation, it cancels out of the volume-to-surface ratio and carries no real reward signal either way. Its near-zero final value looks more like a weakly-explored parameter than a discovery about reactor shape.

### Magnetic field
Close to the accepted range. The agent found a balance that confines the plasma without spending excessive energy generating the field itself.

### Density
Within the same order of magnitude as accepted values, though a little over three times higher than typical.

### Temperature
Roughly two to three times below the generally accepted range. This is likely connected to the inclusion of H-4, which would in theory need less energy to fuse with deuterium than a real D-T reaction would.

### Beta
The agent converged on a value close to 0.05, in line with the accepted range. This parameter appears to have been rewarded correctly.

### Confinement time
Well below the target by roughly two orders of magnitude. Likely a combined result of the small volume, lower-than-ideal temperature, and unrealistic fuel choice all working against sustaining the plasma.

### Total power
The cumulative result of all of the above: a reactor that consistently lost more power than it produced.

## 5. What Worked Well

The magnetic field, density, and beta values all converged to realistic ranges, suggesting the plasma-physics side of the environment is working correctly even where the nuclear side isn't. The later episodes of training also showed noticeably less variance than the early ones. This shows the RL side of the setup was converging properly rather than exploring randomly throughout.

It's also worth noting, even if not an explicit design goal, that being able to trace each poor result back to a specific cause in the code (the shell model, the shape-factor formula) is itself a useful outcome. Any future work on this project now has a clear starting point.

## 6. Conclusion

Overall, the failure of the model traces back to the shell model: an unrealistic fuel choice and an incomplete reaction picture undermined the power output regardless of how well the geometry or plasma physics performed. The parts of the model that don't depend on nuclear structure came out realistic. A better binding-energy model, or simply a lookup table of real nuclear masses, would be a quick fix, but that would defeat the point of the project. The point was to see what an agent could work out given nothing but the underlying physics.

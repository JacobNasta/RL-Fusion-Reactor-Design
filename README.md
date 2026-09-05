# Fusion-RL
This project aimed to create an environment that allowed an agent to search for the parameters needed for a fusion reactor to reach maximum power output. The agent was given a completely open space, covering everything from the fuel used to the reactor's shape and operating conditions. The end result was that the agent had to minimise power loss rather than reach for a net positive output, as the physics driving the rewards was not strong enough. Roughly half of the final parameters the agent settled on matched generally accepted values for a real reactor. Details of the success and limitations of the model can be found in the report.

## Note for the code
Several scaling constants (bremsstrahlung, synchrotron, surface loss, shape factor coefficients) are placeholders chosen to make units and outputs behave sensibly. They are not values derived from first principles. 

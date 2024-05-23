# Transportation-Network-Perimeter-Control

## Files Description:

### Python Files:
- **ppo.py**: PPO algorithm implementation.
- **train.py**: Training code.
- **generate_routes_xml.py**: Configurable script to create a random route XML file.
- **heatmap.py**: Script to visualize the traffic as a heatmap.
- **sumo_env.py**: Wraps and manages the SUMO simulation as a Gym environment. Needs to be installed as a package to use it.

### SUMO Files:
- **network.net.xml**: Network XML configuration.
- **route_file.rou.xml**: Car trips configuration.
- **traffic_lights.tll.xml**: Traffic lights configuration.
- **sim1-5.sumocfg**: Five different SUMO simulations based on different route files.

    

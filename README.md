# Transportation-Network-Perimeter-Control
files description:/n
  python files: 
    ppo.py - ppo algorithm implementation.
    train.py - training code.
    generate_routes_xml.py - configurable script to create a random route xml file.
    heatmap.py - script to visualize the traffic as heatmap.
    sumo_env.py - wrapps and manages the sumo simulation as a gym environment. need to install it as a package to use it.
  sumo files: 
    network.net.xml - network xml configuration.
    route_file.rou.xml - car trips configuration.
    trafic_lights.tll.xml - traffic lights configuration.
    sim1-5.sumocfg - 5 different sumo simulations based on different route file.
    
    

import argparse
import yaml
import numpy as np
from env.carla_connector import CarlaConnector
from env.state_extractor import StateExtractor

def main():
    parser = argparse.ArgumentParser(description="Extract state vectors from CARLA.")
    parser.add_argument("-n", "--iterations", type=int, default=100,
                        help="Number of state vectors to extract")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration YAML")
    parser.add_argument("--output", default="state_vectors.npy",
                        help="Filepath to save extracted state vectors (NumPy binary)")
    args = parser.parse_args()

    # Load configuration (may be used for spawning parameters etc.)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    connector = CarlaConnector(args.config)
    connector.connect()
    world = connector.get_world()

    # TODO: spawn ego vehicle and mock pedestrians here using world and config
    # For now, assume an existing ego vehicle is in the world
    ego_vehicle = world.get_actors().filter("vehicle.*")[0]

    extractor = StateExtractor(connector, config_path=args.config)
    extractor.initialize()

    states = []
    for i in range(args.iterations):
        # Full 40‑dimensional extraction once implemented; fallback to basic
        try:
            state_vector = extractor.extract_state(ego_vehicle)
        except AttributeError:
            # If extract_state is not implemented, use extract_basic_state
            state_vector = extractor.extract_basic_state(ego_vehicle)
        states.append(state_vector)

        # Step the world or tick if in synchronous mode
        if world.get_settings().synchronous_mode:
            world.tick()

    # Save as NumPy array
    states_array = np.vstack(states)
    np.save(args.output, states_array)
    print(f"Saved {len(states)} state vectors to {args.output}")

if __name__ == "__main__":
    main()

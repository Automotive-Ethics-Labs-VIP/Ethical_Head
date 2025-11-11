from env.carla_connector import CarlaConnector
from env.state_extractor import StateExtractor
from validation.validator import Validator
from validation.statistics import Statistics

def main():
    connector = CarlaConnector('config/config.yaml')
    connector.connect()
    world = connector.get_world()

    # assume ego_vehicle is obtained from the world
    ego_vehicle = world.get_actors().filter('vehicle.*')[0]

    extractor = StateExtractor(connector, config_path='config/config.yaml')
    extractor.initialize()

    # extract the full state vector (implement extract_state per issue #4)
    state_vector = extractor.extract_state(ego_vehicle)

    validator = Validator('config/config.yaml')
    stats = Statistics()

    validator.validate(state_vector)  # raises if invalid
    stats.update(state_vector)

    summary = stats.summary()
    stats.save('state_stats.json')
    print(summary)

if __name__ == "__main__":
    main()

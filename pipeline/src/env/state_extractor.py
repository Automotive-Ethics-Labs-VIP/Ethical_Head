import numpy as np
import carla
import yaml
import math
from typing import Optional, Tuple

class StateExtractor:
    """Extract state features from CARLA environment for autonomous driving."""

    def __init__(self, carla_connector, config_path='config/config.yaml'):
        """
        Initialize the state extractor.
        
        Args:
            carla_connector: Instance of CarlaConnector
            config_path: Path to configuration file
        """
        self.connector = carla_connector
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config.get('extraction', {})
        self.world = None
        self.map = None

    def initialize(self);
        """Initialize the CARLA world and map."""
        self.world = self.connector.get_world()
        self.map = self.world.get_map()
    
    def get_ego_velocity(self, ego_vehicle: carla.Vehicle) -> float:
        """
        Returns the speed of the ego vehicle in m/s
        """
        if ego_vehicle is None:
            raise ValueError("Ego vehicle cannot be None")
        
        velocity = ego_vehicle.get_velocity()
        # Calculate magnitude: sqrt(vx^2 + vy^2 + vz^2)
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        return float(speed)
    
    def get_num_passengers(self, ego_vehicle: carla.Vehicle) -> int:
        """
        Returns the number of passengers in the ego vehicle.
        """
        if ego_vehicle is None:
            raise ValueError("Ego vehicle cannot be None")
        
        # return default for now, fix later
        return self.config.get('default_passengers', 1)

    def get_lane_position(self, ego_vehicle: carla.Vehicle) -> float:
        """
        Normalized, 0 is center of lane, -1 is left , 1 is right 
        """
        if ego_vehicle is None:
            raise ValueError("Ego vehicle cannot be None")
        
        if self.map is None:
            raise RuntimeError("Map not initialized. Call initialize() first.")
        
        vehicle_location = ego_vehicle.get_location()

        waypoint = self.map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint is None:
            return 0.0
        
        lane_center = waypoint.transform.location

        dx = vehicle_location.x - lane_center.x
        dy = vehicle_location.y - lane_center.y

        forward = waypoint.transform.get_forward_vector()

        right_x = -forward.y
        right_y = forward.x

        lateral_offset = dx * right_x + dy * right_y
        
        # Normalize by half lane width
        max_offset = self.config.get('max_lane_offset', 1.75)
        normalized_position = lateral_offset / max_offset
        
        normalized_position = max(-1.0, min(1.0, normalized_position))
        

    # def get_velocity_delta():
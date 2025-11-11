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

    def initialize(self):
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
        

    def get_velocity_delta(self, ego_vehicle: carla.Vehicle) -> Optional[float]:
        if ego_vehicle is None:
            raise ValueError("Ego vehicle cannot be None")
        
        if self.world is None:
            raise RuntimeError("World not initialized. Call initialize() first.")
        
        lead_vehicle = self._find_lead_vehicle(ego_vehicle)
        
        if lead_vehicle is None:
            return None
        
        # Get velocities
        ego_velocity = self.get_ego_velocity(ego_vehicle)
        
        lead_vel = lead_vehicle.get_velocity()
        lead_speed = math.sqrt(lead_vel.x**2 + lead_vel.y**2 + lead_vel.z**2)
        
        # Positive if ego is faster
        velocity_delta = ego_velocity - lead_speed
        
        return float(velocity_delta)
    
    def _find_lead_vehicle(self, ego_vehicle: carla.Vehicle) -> Optional[carla.Vehicle]:
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()
        
        # Get all vehicles in the world
        all_vehicles = self.world.get_actors().filter('vehicle.*')
        
        max_distance = self.config.get('max_detection_distance', 50.0)
        detection_angle = self.config.get('detection_angle', 30.0)
        
        closest_vehicle = None
        min_distance = float('inf')
        
        for vehicle in all_vehicles:
            # Skip the ego vehicle itself
            if vehicle.id == ego_vehicle.id:
                continue
            
            vehicle_location = vehicle.get_location()
            
            # Vector from ego to other vehicle
            to_vehicle = carla.Location(
                x=vehicle_location.x - ego_location.x,
                y=vehicle_location.y - ego_location.y,
                z=vehicle_location.z - ego_location.z
            )
            
            # Calculate distance
            distance = math.sqrt(to_vehicle.x**2 + to_vehicle.y**2 + to_vehicle.z**2)
            
            # Skip if too far
            if distance > max_distance:
                continue
            
            # Check if vehicle is in front (dot product > 0)
            dot_product = (
                ego_forward.x * to_vehicle.x +
                ego_forward.y * to_vehicle.y +
                ego_forward.z * to_vehicle.z
            )
            
            if dot_product <= 0:
                # Vehicle is behind or beside
                continue
            
            # Calculate angle between ego forward vector and vector to vehicle
            ego_forward_mag = math.sqrt(
                ego_forward.x**2 + ego_forward.y**2 + ego_forward.z**2
            )
            
            cos_angle = dot_product / (ego_forward_mag * distance)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
            
            # Check if within detection cone
            if angle > detection_angle:
                continue
            
            # Update closest vehicle
            if distance < min_distance:
                min_distance = distance
                closest_vehicle = vehicle
        
        return closest_vehicle
    
    def extract_basic_state(self, ego_vehicle: carla.Vehicle) -> np.ndarray:
        """
        Extract the 4 basic state features and returns array
        """
        if ego_vehicle is None:
            raise ValueError("Ego vehicle cannot be None")
        
        # Extract features
        velocity = self.get_ego_velocity(ego_vehicle)
        passengers = self.get_num_passengers(ego_vehicle)
        lane_pos = self.get_lane_position(ego_vehicle)
        vel_delta = self.get_velocity_delta(ego_vehicle)
        
        # Handle None for velocity_delta (no lead vehicle)
        if vel_delta is None:
            vel_delta = 0.0
        
        # Create state vector
        state = np.array([
            velocity,
            float(passengers),
            lane_pos,
            vel_delta
        ], dtype=np.float32)
        
        return state
    
    def get_state_dict(self, ego_vehicle: carla.Vehicle) -> dict:
        """
        Extract basic state as dictionary for debugging/logging.
        
        Args:
            ego_vehicle: The ego vehicle actor
            
        Returns:
            Dictionary with feature names and values
        """
        velocity = self.get_ego_velocity(ego_vehicle)
        passengers = self.get_num_passengers(ego_vehicle)
        lane_pos = self.get_lane_position(ego_vehicle)
        vel_delta = self.get_velocity_delta(ego_vehicle)
        
        return {
            'velocity_ego': velocity,
            'num_passengers': passengers,
            'lane_position': lane_pos,
            'velocity_delta': vel_delta,
            'has_lead_vehicle': vel_delta is not None
        }
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import carla
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from env.state_extractor import StateExtractor


@pytest.fixture
def mock_connector():
    """Create a mock CARLA connector."""
    connector = Mock()
    connector.get_world = Mock(return_value=Mock())
    connector.get_map = Mock(return_value=Mock())
    return connector


@pytest.fixture
def mock_config_file(tmp_path):
    """Create a temporary config file."""
    config_content = """
carla:
  host: 'localhost'
  port: 2000
  timeout: 10.0
  synchronous_mode: true
  fixed_delta_seconds: 0.05

extraction:
  lane_width: 3.5
  max_lane_offset: 1.75
  max_detection_distance: 50.0
  detection_angle: 30.0
  default_passengers: 1
  max_passengers: 8
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return str(config_path)


@pytest.fixture
def extractor(mock_connector, mock_config_file):
    """Create a StateExtractor instance with mock connector."""
    extractor = StateExtractor(mock_connector, config_path=mock_config_file)
    extractor.initialize()
    return extractor


@pytest.fixture
def mock_ego_vehicle():
    """Create a mock ego vehicle."""
    vehicle = Mock(spec=carla.Vehicle)
    vehicle.id = 1
    
    # Mock velocity
    velocity = Mock()
    velocity.x = 10.0
    velocity.y = 0.0
    velocity.z = 0.0
    vehicle.get_velocity = Mock(return_value=velocity)
    
    # Mock transform and location
    transform = Mock()
    location = Mock()
    location.x = 100.0
    location.y = 50.0
    location.z = 0.0
    transform.location = location
    
    forward_vector = Mock()
    forward_vector.x = 1.0
    forward_vector.y = 0.0
    forward_vector.z = 0.0
    transform.get_forward_vector = Mock(return_value=forward_vector)
    
    vehicle.get_transform = Mock(return_value=transform)
    vehicle.get_location = Mock(return_value=location)
    
    return vehicle


class TestGetEgoVelocity:
    """Test get_ego_velocity method."""
    
    def test_velocity_calculation(self, extractor, mock_ego_vehicle):
        """Test basic velocity calculation."""
        velocity = extractor.get_ego_velocity(mock_ego_vehicle)
        assert velocity == 10.0
    
    def test_velocity_with_multiple_components(self, extractor, mock_ego_vehicle):
        """Test velocity with x, y, z components."""
        vel = Mock()
        vel.x = 3.0
        vel.y = 4.0
        vel.z = 0.0
        mock_ego_vehicle.get_velocity = Mock(return_value=vel)
        
        velocity = extractor.get_ego_velocity(mock_ego_vehicle)
        expected = np.sqrt(3.0**2 + 4.0**2)
        assert abs(velocity - expected) < 1e-6
    
    def test_velocity_stationary(self, extractor, mock_ego_vehicle):
        """Test velocity when vehicle is stationary."""
        vel = Mock()
        vel.x = 0.0
        vel.y = 0.0
        vel.z = 0.0
        mock_ego_vehicle.get_velocity = Mock(return_value=vel)
        
        velocity = extractor.get_ego_velocity(mock_ego_vehicle)
        assert velocity == 0.0
    
    def test_velocity_none_vehicle(self, extractor):
        """Test that None vehicle raises ValueError."""
        with pytest.raises(ValueError, match="Ego vehicle cannot be None"):
            extractor.get_ego_velocity(None)


class TestGetNumPassengers:
    """Test get_num_passengers method."""
    
    def test_default_passengers(self, extractor, mock_ego_vehicle):
        """Test default passenger count."""
        passengers = extractor.get_num_passengers(mock_ego_vehicle)
        assert passengers == 1
    
    def test_passengers_none_vehicle(self, extractor):
        """Test that None vehicle raises ValueError."""
        with pytest.raises(ValueError, match="Ego vehicle cannot be None"):
            extractor.get_num_passengers(None)
    
    def test_passengers_returns_int(self, extractor, mock_ego_vehicle):
        """Test that passenger count is integer."""
        passengers = extractor.get_num_passengers(mock_ego_vehicle)
        assert isinstance(passengers, int)


class TestGetLanePosition:
    """Test get_lane_position method."""
    
    def test_lane_position_center(self, extractor, mock_ego_vehicle):
        """Test vehicle at lane center."""
        # Mock waypoint at same location as vehicle
        waypoint = Mock()
        waypoint_location = Mock()
        waypoint_location.x = 100.0
        waypoint_location.y = 50.0
        waypoint_location.z = 0.0
        
        waypoint_transform = Mock()
        waypoint_transform.location = waypoint_location
        
        forward_vec = Mock()
        forward_vec.x = 1.0
        forward_vec.y = 0.0
        forward_vec.z = 0.0
        waypoint_transform.get_forward_vector = Mock(return_value=forward_vec)
        
        waypoint.transform = waypoint_transform
        extractor.map.get_waypoint = Mock(return_value=waypoint)
        
        lane_pos = extractor.get_lane_position(mock_ego_vehicle)
        assert abs(lane_pos) < 1e-6
    
    def test_lane_position_right_edge(self, extractor, mock_ego_vehicle):
        """Test vehicle at right edge of lane."""
        waypoint = Mock()
        waypoint_location = Mock()
        waypoint_location.x = 100.0
        waypoint_location.y = 50.0
        waypoint_location.z = 0.0
        
        waypoint_transform = Mock()
        waypoint_transform.location = waypoint_location
        
        # Forward is along x-axis, so right is along -y axis
        forward_vec = Mock()
        forward_vec.x = 1.0
        forward_vec.y = 0.0
        forward_vec.z = 0.0
        waypoint_transform.get_forward_vector = Mock(return_value=forward_vec)
        
        waypoint.transform = waypoint_transform
        extractor.map.get_waypoint = Mock(return_value=waypoint)
        
        # Move vehicle to right by max_lane_offset
        vehicle_location = Mock()
        vehicle_location.x = 100.0
        vehicle_location.y = 50.0 - 1.75  # Right edge
        vehicle_location.z = 0.0
        mock_ego_vehicle.get_location = Mock(return_value=vehicle_location)
        
        lane_pos = extractor.get_lane_position(mock_ego_vehicle)
        assert abs(lane_pos - 1.0) < 0.01
    
    def test_lane_position_left_edge(self, extractor, mock_ego_vehicle):
        """Test vehicle at left edge of lane."""
        waypoint = Mock()
        waypoint_location = Mock()
        waypoint_location.x = 100.0
        waypoint_location.y = 50.0
        waypoint_location.z = 0.0
        
        waypoint_transform = Mock()
        waypoint_transform.location = waypoint_location
        
        forward_vec = Mock()
        forward_vec.x = 1.0
        forward_vec.y = 0.0
        forward_vec.z = 0.0
        waypoint_transform.get_forward_vector = Mock(return_value=forward_vec)
        
        waypoint.transform = waypoint_transform
        extractor.map.get_waypoint = Mock(return_value=waypoint)
        
        # Move vehicle to left
        vehicle_location = Mock()
        vehicle_location.x = 100.0
        vehicle_location.y = 50.0 + 1.75  # Left edge
        vehicle_location.z = 0.0
        mock_ego_vehicle.get_location = Mock(return_value=vehicle_location)
        
        lane_pos = extractor.get_lane_position(mock_ego_vehicle)
        assert abs(lane_pos - (-1.0)) < 0.01
    
    def test_lane_position_no_waypoint(self, extractor, mock_ego_vehicle):
        """Test handling when vehicle is not on valid lane."""
        extractor.map.get_waypoint = Mock(return_value=None)
        
        lane_pos = extractor.get_lane_position(mock_ego_vehicle)
        assert lane_pos == 0.0
    
    def test_lane_position_clamping(self, extractor, mock_ego_vehicle):
        """Test that lane position is clamped to [-1, 1]."""
        waypoint = Mock()
        waypoint_location = Mock()
        waypoint_location.x = 100.0
        waypoint_location.y = 50.0
        waypoint_location.z = 0.0
        
        waypoint_transform = Mock()
        waypoint_transform.location = waypoint_location
        
        forward_vec = Mock()
        forward_vec.x = 1.0
        forward_vec.y = 0.0
        forward_vec.z = 0.0
        waypoint_transform.get_forward_vector = Mock(return_value=forward_vec)
        
        waypoint.transform = waypoint_transform
        extractor.map.get_waypoint = Mock(return_value=waypoint)
        
        # Move vehicle far to the right (beyond lane edge)
        vehicle_location = Mock()
        vehicle_location.x = 100.0
        vehicle_location.y = 50.0 - 5.0  # Way off to the right
        vehicle_location.z = 0.0
        mock_ego_vehicle.get_location = Mock(return_value=vehicle_location)
        
        lane_pos = extractor.get_lane_position(mock_ego_vehicle)
        assert lane_pos == 1.0  # Should be clamped
    
    def test_lane_position_none_vehicle(self, extractor):
        """Test that None vehicle raises ValueError."""
        with pytest.raises(ValueError, match="Ego vehicle cannot be None"):
            extractor.get_lane_position(None)


class TestGetVelocityDelta:
    """Test get_velocity_delta method."""
    
    def test_velocity_delta_with_lead_vehicle(self, extractor, mock_ego_vehicle):
        """Test velocity delta when lead vehicle exists."""
        # Create lead vehicle
        lead_vehicle = Mock(spec=carla.Vehicle)
        lead_vehicle.id = 2
        
        lead_vel = Mock()
        lead_vel.x = 8.0
        lead_vel.y = 0.0
        lead_vel.z = 0.0
        lead_vehicle.get_velocity = Mock(return_value=lead_vel)
        
        lead_location = Mock()
        lead_location.x = 120.0  # 20m ahead
        lead_location.y = 50.0
        lead_location.z = 0.0
        lead_vehicle.get_location = Mock(return_value=lead_location)
        
        # Mock world actors
        actors = Mock()
        actors.filter = Mock(return_value=[mock_ego_vehicle, lead_vehicle])
        extractor.world.get_actors = Mock(return_value=actors)
        
        vel_delta = extractor.get_velocity_delta(mock_ego_vehicle)
        
        # Ego is 10 m/s, lead is 8 m/s, so delta should be 2 m/s
        assert abs(vel_delta - 2.0) < 1e-6
    
    def test_velocity_delta_no_lead_vehicle(self, extractor, mock_ego_vehicle):
        """Test velocity delta when no lead vehicle exists."""
        actors = Mock()
        actors.filter = Mock(return_value=[mock_ego_vehicle])
        extractor.world.get_actors = Mock(return_value=actors)
        
        vel_delta = extractor.get_velocity_delta(mock_ego_vehicle)
        assert vel_delta is None
    
    def test_velocity_delta_negative(self, extractor, mock_ego_vehicle):
        """Test velocity delta when lead vehicle is faster."""
        lead_vehicle = Mock(spec=carla.Vehicle)
        lead_vehicle.id = 2
        
        lead_vel = Mock()
        lead_vel.x = 15.0  # Faster than ego
        lead_vel.y = 0.0
        lead_vel.z = 0.0
        lead_vehicle.get_velocity = Mock(return_value=lead_vel)
        
        lead_location = Mock()
        lead_location.x = 120.0
        lead_location.y = 50.0
        lead_location.z = 0.0
        lead_vehicle.get_location = Mock(return_value=lead_location)
        
        actors = Mock()
        actors.filter = Mock(return_value=[mock_ego_vehicle, lead_vehicle])
        extractor.world.get_actors = Mock(return_value=actors)
        
        vel_delta = extractor.get_velocity_delta(mock_ego_vehicle)
        
        # Ego is 10 m/s, lead is 15 m/s, so delta should be -5 m/s
        assert abs(vel_delta - (-5.0)) < 1e-6
    
    def test_velocity_delta_ignores_vehicles_behind(self, extractor, mock_ego_vehicle):
        """Test that vehicles behind ego are ignored."""
        # Vehicle behind ego
        rear_vehicle = Mock(spec=carla.Vehicle)
        rear_vehicle.id = 2
        
        rear_location = Mock()
        rear_location.x = 80.0  # Behind ego
        rear_location.y = 50.0
        rear_location.z = 0.0
        rear_vehicle.get_location = Mock(return_value=rear_location)
        
        actors = Mock()
        actors.filter = Mock(return_value=[mock_ego_vehicle, rear_vehicle])
        extractor.world.get_actors = Mock(return_value=actors)
        
        vel_delta = extractor.get_velocity_delta(mock_ego_vehicle)
        assert vel_delta is None
    
    def test_velocity_delta_ignores_far_vehicles(self, extractor, mock_ego_vehicle):
        """Test that vehicles beyond max distance are ignored."""
        far_vehicle = Mock(spec=carla.Vehicle)
        far_vehicle.id = 2
        
        far_location = Mock()
        far_location.x = 200.0  # 100m ahead (> max_detection_distance)
        far_location.y = 50.0
        far_location.z = 0.0
        far_vehicle.get_location = Mock(return_value=far_location)
        
        actors = Mock()
        actors.filter = Mock(return_value=[mock_ego_vehicle, far_vehicle])
        extractor.world.get_actors = Mock(return_value=actors)
        
        vel_delta = extractor.get_velocity_delta(mock_ego_vehicle)
        assert vel_delta is None
    
    def test_velocity_delta_none_vehicle(self, extractor):
        """Test that None vehicle raises ValueError."""
        with pytest.raises(ValueError, match="Ego vehicle cannot be None"):
            extractor.get_velocity_delta(None)


class TestExtractBasicState:
    """Test extract_basic_state method."""
    
    def test_extract_basic_state_shape(self, extractor, mock_ego_vehicle):
        """Test that state vector has correct shape."""
        # Setup mocks
        waypoint = Mock()
        waypoint_location = Mock()
        waypoint_location.x = 100.0
        waypoint_location.y = 50.0
        waypoint_location.z = 0.0
        waypoint_transform = Mock()
        waypoint_transform.location = waypoint_location
        forward_vec = Mock()
        forward_vec.x = 1.0
        forward_vec.y = 0.0
        forward_vec.z = 0.0
        waypoint_transform.get_forward_vector = Mock(return_value=forward_vec)
        waypoint.transform = waypoint_transform
        extractor.map.get_waypoint = Mock(return_value=waypoint)
        
        actors = Mock()
        actors.filter = Mock(return_value=[mock_ego_vehicle])
        extractor.world.get_actors = Mock(return_value=actors)
        
        state = extractor.extract_basic_state(mock_ego_vehicle)
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (4,)
    
    def test_extract_basic_state_dtype(self, extractor, mock_ego_vehicle):
        """Test that state vector has correct dtype."""
        waypoint = Mock()
        waypoint_location = Mock()
        waypoint_location.x = 100.0
        waypoint_location.y = 50.0
        waypoint_location.z = 0.0
        waypoint_transform = Mock()
        waypoint_transform.location = waypoint_location
        forward_vec = Mock()
        forward_vec.x = 1.0
        forward_vec.y = 0.0
        forward_vec.z = 0.0
        waypoint_transform.get_forward_vector = Mock(return_value=forward_vec)
        waypoint.transform = waypoint_transform
        extractor.map.get_waypoint = Mock(return_value=waypoint)
        
        actors = Mock()
        actors.filter = Mock(return_value=[mock_ego_vehicle])
        extractor.world.get_actors = Mock(return_value=actors)
        
        state = extractor.extract_basic_state(mock_ego_vehicle)
        
        assert state.dtype == np.float32
    
    def test_extract_basic_state_values(self, extractor, mock_ego_vehicle):
        """Test that state vector contains correct values."""
        waypoint = Mock()
        waypoint_location = Mock()
        waypoint_location.x = 100.0
        waypoint_location.y = 50.0
        waypoint_location.z = 0.0
        waypoint_transform = Mock()
        waypoint_transform.location = waypoint_location
        forward_vec = Mock()
        forward_vec.x = 1.0
        forward_vec.y = 0.0
        forward_vec.z = 0.0
        waypoint_transform.get_forward_vector = Mock(return_value=forward_vec)
        waypoint.transform = waypoint_transform
        extractor.map.get_waypoint = Mock(return_value=waypoint)
        
        actors = Mock()
        actors.filter = Mock(return_value=[mock_ego_vehicle])
        extractor.world.get_actors = Mock(return_value=actors)
        
        state = extractor.extract_basic_state(mock_ego_vehicle)
        
        # velocity_ego = 10.0 (from mock)
        assert abs(state[0] - 10.0) < 1e-6
        # num_passengers = 1 (default)
        assert state[1] == 1.0
        # lane_position = 0.0 (center)
        assert abs(state[2]) < 1e-6
        # velocity_delta = 0.0 (no lead vehicle)
        assert state[3] == 0.0
    
    def test_extract_basic_state_none_velocity_delta(self, extractor, mock_ego_vehicle):
        """Test that None velocity_delta is converted to 0.0."""
        waypoint = Mock()
        waypoint_location = Mock()
        waypoint_location.x = 100.0
        waypoint_location.y = 50.0
        waypoint_location.z = 0.0
        waypoint_transform = Mock()
        waypoint_transform.location = waypoint_location
        forward_vec = Mock()
        forward_vec.x = 1.0
        forward_vec.y = 0.0
        forward_vec.z = 0.0
        waypoint_transform.get_forward_vector = Mock(return_value=forward_vec)
        waypoint.transform = waypoint_transform
        extractor.map.get_waypoint = Mock(return_value=waypoint)
        
        actors = Mock()
        actors.filter = Mock(return_value=[mock_ego_vehicle])
        extractor.world.get_actors = Mock(return_value=actors)
        
        state = extractor.extract_basic_state(mock_ego_vehicle)
        
        # velocity_delta should be 0.0 when no lead vehicle
        assert state[3] == 0.0
    
    def test_extract_basic_state_none_vehicle(self, extractor):
        """Test that None vehicle raises ValueError."""
        with pytest.raises(ValueError, match="Ego vehicle cannot be None"):
            extractor.extract_basic_state(None)


class TestGetStateDict:
    """Test get_state_dict method."""
    
    def test_state_dict_structure(self, extractor, mock_ego_vehicle):
        """Test that state dict has correct structure."""
        waypoint = Mock()
        waypoint_location = Mock()
        waypoint_location.x = 100.0
        waypoint_location.y = 50.0
        waypoint_location.z = 0.0
        waypoint_transform = Mock()
        waypoint_transform.location = waypoint_location
        forward_vec = Mock()
        forward_vec.x = 1.0
        forward_vec.y = 0.0
        forward_vec.z = 0.0
        waypoint_transform.get_forward_vector = Mock(return_value=forward_vec)
        waypoint.transform = waypoint_transform
        extractor.map.get_waypoint = Mock(return_value=waypoint)
        
        actors = Mock()
        actors.filter = Mock(return_value=[mock_ego_vehicle])
        extractor.world.get_actors = Mock(return_value=actors)
        
        state_dict = extractor.get_state_dict(mock_ego_vehicle)
        
        assert 'velocity_ego' in state_dict
        assert 'num_passengers' in state_dict
        assert 'lane_position' in state_dict
        assert 'velocity_delta' in state_dict
        assert 'has_lead_vehicle' in state_dict
    
    def test_state_dict_has_lead_vehicle_flag(self, extractor, mock_ego_vehicle):
        """Test has_lead_vehicle flag is correct."""
        waypoint = Mock()
        waypoint_location = Mock()
        waypoint_location.x = 100.0
        waypoint_location.y = 50.0
        waypoint_location.z = 0.0
        waypoint_transform = Mock()
        waypoint_transform.location = waypoint_location
        forward_vec = Mock()
        forward_vec.x = 1.0
        forward_vec.y = 0.0
        forward_vec.z = 0.0
        waypoint_transform.get_forward_vector = Mock(return_value=forward_vec)
        waypoint.transform = waypoint_transform
        extractor.map.get_waypoint = Mock(return_value=waypoint)
        
        actors = Mock()
        actors.filter = Mock(return_value=[mock_ego_vehicle])
        extractor.world.get_actors = Mock(return_value=actors)
        
        state_dict = extractor.get_state_dict(mock_ego_vehicle)
        
        # No lead vehicle, so should be False
        assert state_dict['has_lead_vehicle'] is False
        assert state_dict['velocity_delta'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

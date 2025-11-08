import pytest
import sys
sys.path.append('src')

from environment.carla_connector import CarlaConnector


def test_connection():
    """Test that client connects to CARLA server."""
    connector = CarlaConnector()
    
    try:
        connector.connect()
        assert connector.client is not None
        print("✓ Client connected successfully")
    except ConnectionError as e:
        pytest.skip(f"CARLA server not running: {e}")


def test_world_access():
    """Test that world object is accessible."""
    connector = CarlaConnector()
    
    try:
        connector.connect()
        world = connector.get_world()
        assert world is not None
        print("✓ World object accessible")
    except ConnectionError as e:
        pytest.skip(f"CARLA server not running: {e}")


def test_blueprint_library():
    """Test that blueprint library contains pedestrian blueprints."""
    connector = CarlaConnector()
    
    try:
        connector.connect()
        bp_lib = connector.get_blueprint_library()
        pedestrians = bp_lib.filter('walker.pedestrian.*')
        assert len(pedestrians) > 0
        print(f"✓ Found {len(pedestrians)} pedestrian blueprints")
    except ConnectionError as e:
        pytest.skip(f"CARLA server not running: {e}")


def test_map_spawn_points():
    """Test that map object returns spawn points."""
    connector = CarlaConnector()
    
    try:
        connector.connect()
        map_obj = connector.get_map()
        spawn_points = map_obj.get_spawn_points()
        assert len(spawn_points) > 0
        print(f"✓ Found {len(spawn_points)} spawn points")
    except ConnectionError as e:
        pytest.skip(f"CARLA server not running: {e}")


def test_synchronous_mode():
    """Test that synchronous mode is enabled."""
    connector = CarlaConnector()
    
    try:
        connector.connect()
        world = connector.get_world()
        settings = world.get_settings()
        assert settings.synchronous_mode == True
        assert settings.fixed_delta_seconds == 0.05
        print("✓ Synchronous mode configured correctly")
    except ConnectionError as e:
        pytest.skip(f"CARLA server not running: {e}")


def test_connection_error_handling():
    """Test that connection errors are caught properly."""
    connector = CarlaConnector()
    connector.config['port'] = 9999  # Invalid port
    
    with pytest.raises(ConnectionError):
        connector.connect()
    print("✓ Connection errors handled properly")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

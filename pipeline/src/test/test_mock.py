"""Tests for the mock pedestrian generator."""

import os
import random
import sys
import types
from collections import Counter
from typing import List

from unittest.mock import Mock

import pytest

if "carla" not in sys.modules:
    carla_module = types.ModuleType("carla")

    class Actor:  # pragma: no cover - simple stub for tests
        id: int

    class Location:
        def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
            self.x = x
            self.y = y
            self.z = z

    class Rotation:
        def __init__(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0):
            self.pitch = pitch
            self.yaw = yaw
            self.roll = roll

    class Transform:
        def __init__(self, location: "Location", rotation: "Rotation" = None) -> None:
            self.location = location
            self.rotation = rotation or Rotation()

    carla_module.Actor = Actor
    carla_module.Location = Location
    carla_module.Rotation = Rotation
    carla_module.Transform = Transform

    sys.modules["carla"] = carla_module

import carla

# Ensure the src directory is importable when running tests directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.mock_pedestrian import MockPedestrianGenerator, PedestrianAttributes


@pytest.fixture
def temp_config(tmp_path):
    config_content = """
carla:
  host: 'localhost'
  port: 2000
  timeout: 10.0
  synchronous_mode: true
  fixed_delta_seconds: 0.05

mock_pedestrians:
  probabilities:
    is_child: 0.2
    is_elderly: 0.1
    is_pregnant: 0.05
    is_disabled: 0.15
"""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return str(config_path)


@pytest.fixture
def blueprint_library():
    library = Mock()
    library.filter.return_value = [Mock(name="walker1"), Mock(name="walker2")]
    return library


@pytest.fixture
def mock_world():
    world = Mock()

    def spawn_actor(blueprint, transform):
        actor = Mock(spec=carla.Actor)
        actor.id = spawn_actor.counter
        spawn_actor.counter += 1
        return actor

    spawn_actor.counter = 1
    world.spawn_actor.side_effect = spawn_actor

    location = carla.Location(x=0.0, y=0.0, z=0.0)
    world.get_random_location_from_navigation.return_value = location
    return world


def create_spawn_points(count: int) -> List[carla.Transform]:
    return [
        carla.Transform(carla.Location(x=float(index), y=0.0, z=0.0))
        for index in range(count)
    ]


class TestMockPedestrianGenerator:
    def test_spawn_pedestrians(self, mock_world, blueprint_library, temp_config):
        generator = MockPedestrianGenerator(
            mock_world,
            blueprint_library,
            config_path=temp_config,
            probabilities={
                "is_child": 0.2,
                "is_elderly": 0.1,
                "is_pregnant": 0.05,
                "is_disabled": 0.15,
            },
            rng=None,
        )

        spawn_points = create_spawn_points(3)
        spawned = generator.spawn_pedestrians(3, spawn_points)

        assert len(spawned) == 3
        assert mock_world.spawn_actor.call_count == 3
        for actor, attributes in spawned:
            assert isinstance(attributes, PedestrianAttributes)
            assert generator.get_attributes(actor) == attributes

    def test_attribute_assignment(self, mock_world, blueprint_library, temp_config):
        generator = MockPedestrianGenerator(
            mock_world,
            blueprint_library,
            config_path=temp_config,
            probabilities={
                "is_child": 0.2,
                "is_elderly": 0.1,
                "is_pregnant": 0.05,
                "is_disabled": 0.15,
            },
            rng=random.Random(42),
        )

        spawn_points = create_spawn_points(2)
        spawned = generator.spawn_pedestrians(2, spawn_points)

        for _, attributes in spawned:
            assert isinstance(attributes.is_child, bool)
            assert isinstance(attributes.is_elderly, bool)
            assert isinstance(attributes.is_pregnant, bool)
            assert isinstance(attributes.is_disabled, bool)

    def test_attribute_distribution(self, mock_world, blueprint_library, temp_config):
        rng = random.Random(7)
        generator = MockPedestrianGenerator(
            mock_world,
            blueprint_library,
            config_path=temp_config,
            probabilities={
                "is_child": 0.2,
                "is_elderly": 0.1,
                "is_pregnant": 0.05,
                "is_disabled": 0.15,
            },
            rng=rng,
        )

        count = 500
        spawn_points = create_spawn_points(count)
        spawned = generator.spawn_pedestrians(count, spawn_points)

        totals = Counter()
        for _, attributes in spawned:
            totals["is_child"] += int(attributes.is_child)
            totals["is_elderly"] += int(attributes.is_elderly)
            totals["is_pregnant"] += int(attributes.is_pregnant)
            totals["is_disabled"] += int(attributes.is_disabled)

        for key, expected_probability in {
            "is_child": 0.2,
            "is_elderly": 0.1,
            "is_pregnant": 0.05,
            "is_disabled": 0.15,
        }.items():
            observed = totals[key] / count
            assert abs(observed - expected_probability) < 0.05


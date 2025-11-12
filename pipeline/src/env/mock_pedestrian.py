"""Utilities for generating mock pedestrians with random attributes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from importlib import import_module
from typing import Dict, Iterable, List, Optional, Tuple, Union

import carla


@dataclass(frozen=True)
class PedestrianAttributes:
    """Simple container for mock pedestrian attributes."""

    is_child: bool
    is_elderly: bool
    is_pregnant: bool
    is_disabled: bool


class MockPedestrianGenerator:
    """Generate pedestrians with mock attributes for testing and prototyping."""

    DEFAULT_PROBABILITIES = {
        "is_child": 0.1,
        "is_elderly": 0.1,
        "is_pregnant": 0.05,
        "is_disabled": 0.05,
    }

    def __init__(
        self,
        world: carla.World,
        blueprint_library: carla.BlueprintLibrary,
        config_path: str = "config/config.yaml",
        probabilities: Optional[Dict[str, float]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        if world is None:
            raise ValueError("world must not be None")
        if blueprint_library is None:
            raise ValueError("blueprint_library must not be None")

        self._world = world
        self._blueprint_library = blueprint_library
        self._config_path = config_path
        self._rng = rng or random.Random()

        if probabilities is None:
            self._probabilities = self._load_probabilities(config_path)
        else:
            self._probabilities = self._validate_probabilities(probabilities)
        self._attributes_by_actor_id: Dict[int, PedestrianAttributes] = {}

    @staticmethod
    def _load_probabilities(config_path: str) -> Dict[str, float]:
        yaml = import_module("yaml")
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}

        section = config.get("mock_pedestrians", {})
        probs = section.get("probabilities", {})

        return MockPedestrianGenerator._validate_probabilities(probs)

    @staticmethod
    def _validate_probabilities(probs: Optional[Dict[str, float]]) -> Dict[str, float]:
        merged = dict(MockPedestrianGenerator.DEFAULT_PROBABILITIES)
        if probs:
            for key, value in probs.items():
                if key in merged:
                    merged[key] = float(value)

        for key, value in merged.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Probability for {key} must be between 0 and 1.")

        return merged

    def spawn_pedestrians(
        self,
        count: int,
        spawn_points: Optional[Iterable[carla.Transform]] = None,
    ) -> List[Tuple[carla.Actor, PedestrianAttributes]]:
        if count <= 0:
            return []

        walker_blueprints = self._blueprint_library.filter("walker.pedestrian.*")
        if not walker_blueprints:
            raise RuntimeError("No walker blueprints available for spawning pedestrians.")

        spawn_points_list: List[carla.Transform] = list(spawn_points or [])
        spawned: List[Tuple[carla.Actor, PedestrianAttributes]] = []

        for index in range(count):
            blueprint = self._rng.choice(walker_blueprints)
            transform = self._resolve_spawn_transform(index, spawn_points_list)

            actor = self._world.spawn_actor(blueprint, transform)
            attributes = self._generate_attributes()

            self._attributes_by_actor_id[actor.id] = attributes
            spawned.append((actor, attributes))

        return spawned

    def _resolve_spawn_transform(
        self, index: int, spawn_points: List[carla.Transform]
    ) -> carla.Transform:
        if index < len(spawn_points):
            return spawn_points[index]

        location = self._world.get_random_location_from_navigation()
        if location is None:
            raise RuntimeError("Unable to find a spawn location for pedestrian.")

        return carla.Transform(location)

    def _generate_attributes(self) -> PedestrianAttributes:
        return PedestrianAttributes(
            is_child=self._rng.random() < self._probabilities["is_child"],
            is_elderly=self._rng.random() < self._probabilities["is_elderly"],
            is_pregnant=self._rng.random() < self._probabilities["is_pregnant"],
            is_disabled=self._rng.random() < self._probabilities["is_disabled"],
        )

    def get_attributes(
        self, actor_or_id: Union[carla.Actor, int]
    ) -> Optional[PedestrianAttributes]:
        actor_id = actor_or_id if isinstance(actor_or_id, int) else actor_or_id.id
        return self._attributes_by_actor_id.get(actor_id)

    def clear(self) -> None:
        """Forget all stored pedestrian attributes."""

        self._attributes_by_actor_id.clear()


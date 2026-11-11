
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.detection.motion_detector import MotionRegion


@dataclass
class TrackedObject:
    object_id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    disappeared: int = 0


class ObjectTracker:
    def __init__(
        self,
        max_disappeared: int = 10,
        max_distance: float = 50.0,
    ):
        if max_disappeared < 0:
            raise ValueError("max_disappeared cannot be negative")

        if max_distance <= 0:
            raise ValueError("max_distance must be positive")

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.next_object_id = 1
        self.objects: Dict[int, TrackedObject] = {}

    def _register(self, region: MotionRegion):
        object_id = self.next_object_id
        self.next_object_id += 1

        self.objects[object_id] = TrackedObject(
            object_id=object_id,
            centroid=region.center,
            bbox=region.bbox,
        )

    def _deregister(self, object_id: int):
        del self.objects[object_id]

    def update(
        self,
        regions: List[MotionRegion],
    ) -> Dict[int, TrackedObject]:

        if not regions:
            for object_id in list(self.objects.keys()):
                self.objects[object_id].disappeared += 1

                if (
                    self.objects[object_id].disappeared
                    > self.max_disappeared
                ):
                    self._deregister(object_id)

            return self.objects.copy()

        if not self.objects:
            for region in regions:
                self._register(region)

            return self.objects.copy()

        object_ids = list(self.objects.keys())

        old_centroids = np.array(
            [self.objects[object_id].centroid for object_id in object_ids],
            dtype=np.float32,
        )

        new_centroids = np.array(
            [region.center for region in regions],
            dtype=np.float32,
        )

        distances = np.linalg.norm(
            old_centroids[:, np.newaxis] - new_centroids[np.newaxis, :],
            axis=2,
        )

        # Build all possible matches, nearest first.
        candidates = [
            (distances[i, j], i, j)
            for i in range(len(object_ids))
            for j in range(len(regions))
        ]

        candidates.sort(key=lambda item: item[0])

        matched_objects = set()
        matched_regions = set()

        for distance, object_index, region_index in candidates:
            if distance > self.max_distance:
                break

            if object_index in matched_objects:
                continue

            if region_index in matched_regions:
                continue

            object_id = object_ids[object_index]
            region = regions[region_index]

            tracked_object = self.objects[object_id]

            tracked_object.centroid = region.center
            tracked_object.bbox = region.bbox
            tracked_object.disappeared = 0

            matched_objects.add(object_index)
            matched_regions.add(region_index)

        # Handle objects that were not matched.
        for index, object_id in enumerate(object_ids):
            if index not in matched_objects:
                self.objects[object_id].disappeared += 1

                if (
                    self.objects[object_id].disappeared
                    > self.max_disappeared
                ):
                    self._deregister(object_id)

        # Register newly detected objects.
        for index, region in enumerate(regions):
            if index not in matched_regions:
                self._register(region)

        return self.objects.copy()

    def reset(self):
        self.objects.clear()
        self.next_object_id = 1

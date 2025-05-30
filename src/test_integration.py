from typing import Dict

import pytest
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ServiceConfig
from viam.services.vision import Vision

from src.re_id_tracker import ReIDObjetcTracker
from tests.fake_camera import FakeCamera
from tests.utils import get_config

CAMERA_NAME = "fake-camera"

PASSING_PROPERTIES = Vision.Properties(
    classifications_supported=True,
    detections_supported=True,
    object_point_clouds_supported=False,
)

# TODO: test detectors
# MIN_CONFIDENCE_PASSING = 0.8

WORKING_CONFIG_DICT = {
    "camera_name": CAMERA_NAME,
    "path_to_known_persons": "./known_persons",
    "path_to_database": "./something.db",  # TODO: make it pure
    "_start_background_loop": False,
    # "crop_region": {
    #     "y1_rel": 0.5,
    #     "x2_rel": 1,
    #     "y2_rel": 1,
    #     "x1_rel": 0.5,
    # },
}


IMG_PATH = "./tests/alex"


def get_vision_service(config_dict: Dict, reconfigure=True):
    cam = FakeCamera(CAMERA_NAME, img_path=IMG_PATH, use_ring_buffer=True)
    camera_name = cam.get_resource_name(CAMERA_NAME)
    cfg = get_config(config_dict)
    service = ReIDObjetcTracker("test")
    service.validate_config(cfg)

    if reconfigure:
        service.reconfigure(cfg, dependencies={camera_name: cam})
    return service


class TestFaceReId:
    @pytest.mark.asyncio
    async def test_person_reid(self):
        service = get_vision_service(WORKING_CONFIG_DICT, reconfigure=True)
        service.tracker.compute_known_persons_embeddings()
        assert len(service.tracker.labeled_person_embeddings["robin"]) == 1
        assert len(service.tracker.labeled_person_embeddings["alex"]) == 2

        img = await service.tracker.get_and_decode_img()
        service.tracker.last_image = img
        # brand new tracks are not classified as new person but as track candidate
        for i in range(service.tracker.minimum_track_persistance + 1):
            service.tracker.update(img=img)
            assert len(service.tracker.tracks) == 0
            assert len(service.tracker.track_candidates) == 1
            assert service.tracker.track_candidates[0].persistence == i
        service.tracker.update(img=img)

        # now track candidate should have been promoted to track
        assert len(service.tracker.tracks) == 1
        assert len(service.tracker.track_candidates) == 0
        for track in service.tracker.tracks.values():
            assert track.label_from_reid == "alex"
        detections = await service.get_detections_from_camera(
            camera_name=CAMERA_NAME, extra={}, timeout=10
        )
        print(detections)

        await service.close()

    @pytest.mark.asyncio
    async def test_get_properties(self):
        service = ReIDObjetcTracker("test")
        p = await service.get_properties()
        assert p == PASSING_PROPERTIES


if __name__ == "__main__":
    # Run all tests with pytest
    pytest.main(
        ["-xvs", __file__]
    )  # verbose, stop after first failure, don't capture output

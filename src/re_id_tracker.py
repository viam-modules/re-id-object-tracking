"""
This module provides a Viam Vision Service module
to perform face Re-Id.
"""

from asyncio import create_task
import asyncio
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence

from typing_extensions import Self
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.video import CameraMimeType, ViamImage
from viam.module.types import Reconfigurable
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes

from src.config.config import ReIDObjetcTrackerConfig
from src.tracker.tracker import Tracker

LOGGER = getLogger(__name__)


class ReIDObjetcTracker(Vision, Reconfigurable):
    """ReIDTracker is a subclass a Viam Vision Service"""

    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam", "vision"), "re-id-object-tracker"
    )

    def __init__(self, name: str):
        super().__init__(name=name)
        self.camera = None
        self.tracker = None

    @classmethod
    def new_service(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """returns new vision service"""
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    # Validates JSON Configuration
    @classmethod
    def validate_config(cls, config: ServiceConfig) -> Sequence[str]:
        """Validate config and returns a list of dependencies."""
        camera_name = config.attributes.fields["camera_name"].string_value
        _ = ReIDObjetcTrackerConfig(config)
        return [camera_name]

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        self.camera_name = config.attributes.fields["camera_name"].string_value
        self.camera = dependencies[Camera.get_resource_name(self.camera_name)]

        re_id_tracker_cfg = ReIDObjetcTrackerConfig(config)
        if self.tracker is not None:
            create_task(self.stop_and_get_new_tracker(re_id_tracker_cfg))
        else:
            self.tracker = Tracker(re_id_tracker_cfg, camera=self.camera)
            self.tracker.start()

    async def stop_and_get_new_tracker(self, re_id_tracker_cfg):
        await self.tracker.stop()
        self.tracker = Tracker(re_id_tracker_cfg, camera=self.camera)
        self.tracker.start()

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        return Vision.Properties(
            classifications_supported=True,
            detections_supported=True,
            object_point_clouds_supported=False,
        )

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ):
        if not camera_name == self.camera_name:
            raise ValueError(
                "The camera_name %s doesn't match the camera_name configured for the tracker: %s."
                % (camera_name, self.camera_name)
            )
        img = None
        if return_image:
            img = await self.camera.get_image(mime_type=CameraMimeType.JPEG)

        classifications = None
        if return_classifications:
            if await self.tracker.is_new_object_detected():
                classifications = [
                    Classification(class_name="new_object_detected", confidence=1)
                ]

        detections = None
        if return_detections:
            detections = self.tracker.get_current_detections()

        # do_cmd_res = None
        # if extra is not None:
        #     do_cmd_res = await self.do_command(extra)

        return CaptureAllResult(
            image=img, classifications=classifications, detections=detections
        )

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[PointCloudObject]:
        raise NotImplementedError

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Mapping[str, Any],
        timeout: float,
    ) -> List[Detection]:
        return NotImplementedError

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        return NotImplementedError

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        if not camera_name == self.camera_name:
            raise ValueError(
                "The camera_name doesn't match the camera_name configured for the tracker."
            )
        if await self.tracker.is_new_object_detected():
            return [Classification(class_name="new_object_detected", confidence=1)]

    async def get_detections_from_camera(
        self, camera_name: str, *, extra: Mapping[str, Any], timeout: float
    ) -> List[Detection]:
        if not camera_name == self.camera_name:
            raise ValueError(
                "The camera_name doesn't match the camera_name configured for the tracker."
            )
        return self.tracker.get_current_detections()

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        do_command_output = {}
        relabel_cmd = command.get("relabel", None)
        if relabel_cmd is not None:
            do_command_output["relabel"] = self.tracker.relabel_tracks(relabel_cmd)

        add_cmd = command.get("add", None)
        if add_cmd is not None:
            do_command_output["add"] = self.tracker.add_labeled_embedding(add_cmd)

        delete_cmd = command.get("delete", None)
        if delete_cmd is not None:
            do_command_output["delete"] = self.tracker.delete_labeled_embedding(
                delete_cmd
            )

        list_cmd = command.get("list", None)
        if list_cmd:
            do_command_output["list"] = self.tracker.list_objects()

        list_current_cmd = command.get("list_current", None)
        if list_current_cmd:
            do_command_output["list_current"] = self.tracker.list_current()

        recompute_embeddings = command.get("recompute_embeddings", None)
        if recompute_embeddings:
            do_command_output["recompute_embeddings"] = (
                self.tracker.recompute_embeddings()
            )

        return do_command_output

    async def close(self):
        """Safely shut down the resource and prevent further use.

        Close must be idempotent. Later configuration may allow a resource to be "open" again.
        If a resource does not want or need a close function, it is assumed that the resource does not need to return errors when future
        non-Close methods are called.

        ::

            await component.close()

        """
        await self.tracker.stop()  # TODO: ask Naveed
        await super().close()

        return

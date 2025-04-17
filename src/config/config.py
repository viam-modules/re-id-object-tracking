from viam.proto.app.robot import ServiceConfig

from src.config.attribute import (
    BoolAttribute,
    DictAttribute,
    FloatAttribute,
    IntAttribute,
    StringAttribute,
)


class TrackerConfig:
    def __init__(self, config: "ServiceConfig"):
        self.re_id_threshold = FloatAttribute(
            field_name="re_id_threshold",
            min_value=0,
            default_value=0.3,
        ).validate(config)

        self.min_track_persistence = IntAttribute(
            field_name="min_track_persistence",
            min_value=0,
            default_value=4,
        ).validate(config)
        self.lambda_value = FloatAttribute(
            field_name="lambda_value",
            min_value=0,
            max_value=1,
            default_value=0.95,
        ).validate(config)
        self.max_age_track = IntAttribute(
            field_name="max_age_track",
            min_value=0,
            max_value=100000,
            default_value=1000,
        ).validate(config)
        self.min_distance_threshold = FloatAttribute(
            field_name="min_distance_threshold",
            min_value=0,
            max_value=1,
            default_value=0.3,
        ).validate(config)
        self.feature_distance_metric = StringAttribute(
            field_name="feature_distance_metric",
            default_value="cosine",
            allowlist=["cosine", "euclidean"],
        ).validate(config)

        self.max_frequency = FloatAttribute(
            field_name="max_frequency_hz",
            default_value=10,
            min_value=0.1,
            max_value=100,
        ).validate(config)

        self.cooldown_period = FloatAttribute(
            field_name="cooldown_period_s",
            default_value=5,
            min_value=0,
        ).validate(config)

        self.start_fresh = BoolAttribute(
            field_name="start_fresh",
            default_value=False,
        ).validate(config)

        self.save_to_db = BoolAttribute(
            field_name="save_to_db",
            default_value=True,
        ).validate(config)

        self._start_background_loop = BoolAttribute(
            field_name="_start_background_loop", default_value=True
        ).validate(config)

        self.path_to_known_persons = StringAttribute(
            field_name="path_to_known_persons",
            default_value=None,
        ).validate(config)

        self.crop_region = DictAttribute(
            field_name="crop_region",
            default_value=None,
            fields=[
                FloatAttribute(field_name="x1_rel", min_value=0, max_value=1),
                FloatAttribute(field_name="y1_rel", min_value=0, max_value=1),
                FloatAttribute(field_name="x2_rel", min_value=0, max_value=1),
                FloatAttribute(field_name="y2_rel", min_value=0, max_value=1),
            ],
        ).validate(config)


class DetectorConfig:
    def __init__(self, config: "ServiceConfig"):
        self.model_name = StringAttribute(
            field_name="detector_model_name",
            default_value="fasterrcnn_mobilenet_v3_large_320_fpn",
            allowlist=[
                "fasterrcnn_mobilenet_v3_large_320_fpn",  # low resolution model (best effort to resize smallest edge to 320)
                "fasterrcnn_mobilenet_v3_large_fpn",  #  higher resolution model (best effort to resize smallest edge to 800)
            ],
        ).validate(config)
        self.threshold = FloatAttribute(
            field_name="detection_threshold",
            min_value=0.0,
            max_value=1.0,
            default_value=0.95,
        ).validate(config)
        self.device = StringAttribute(
            field_name="detector_device",
            default_value="cpu",
            allowlist=["cpu", "cuda"],
        ).validate(config)

        # TODO: add usage for torchvision
        self.max_results = IntAttribute(
            field_name="detection_max_detection_results",
            default_value=5,
            min_value=1,
        ).validate(config)

        self._enable_debug_tools = BoolAttribute(
            field_name="_enable_debug_tools", default_value=False
        ).validate(config)

        self._path_to_debug_directory = StringAttribute(
            field_name="_path_to_debug_directory",
            default_value=None,
        ).validate(config)

        self._max_size_debug_directory = IntAttribute(
            field_name="_max_size_debug_directory", default_value=200
        ).validate(config)


class FaceIdConfig:
    def __init__(self, config: ServiceConfig):
        self.path_to_known_faces = StringAttribute(
            field_name="path_to_known_faces",
            default_value=None,
        ).validate(config)

        self.device = StringAttribute(
            field_name="face_detector_device",
            default_value="cpu",
            allowlist=["cpu", "cuda"],
        ).validate(config)

        self.detector = StringAttribute(
            field_name="face_detector_model",
            default_value="ultraface_version-RFB-320-int8",
        ).validate(config)

        self.detector_threshold = FloatAttribute(
            field_name="face_detection_threshold",
            min_value=0.0,
            max_value=1.0,
            default_value=0.9,
        ).validate(config)

        self.feature_extractor = StringAttribute(
            field_name="face_feature_extractor_model",
            default_value="facenet",
        ).validate(config)

        self.cosine_id_threshold = FloatAttribute(
            field_name="cosine_id_threshold",
            min_value=0.0,
            max_value=1.0,
            default_value=0.3,
        ).validate(config)

        self.euclidean_id_threshold = FloatAttribute(
            field_name="euclidean_id_threshold",
            min_value=0.0,
            max_value=1.0,
            default_value=0.9,
        ).validate(config)


class FeatureEncoderConfig:
    def __init__(self, config: ServiceConfig):
        self.feature_extractor_name = StringAttribute(
            field_name="feature_extractor_model",
            default_value="osnet_ain_x1_0",
            allowlist=["osnet_x0_25", "osnet_ain_x1_0"],
        ).validate(config)

        self.device = StringAttribute(
            field_name="feature_encoder_device",
            default_value="cuda",
            allowlist=["cpu", "cuda"],
        ).validate(config)


class TracksManagerConfig:
    def __init__(self, config: "ServiceConfig"):
        self.path_to_db = StringAttribute(
            field_name="path_to_database", required=True
        ).validate(config)

        self.save_period = IntAttribute(
            field_name="save_period",
            default_value=20,
        ).validate(config)


class ReIDObjetcTrackerConfig:
    def __init__(self, config: ServiceConfig):
        self.config = config

        self.tracker_config = TrackerConfig(config)
        self.detector_config = DetectorConfig(config)
        self.encoder_config = FeatureEncoderConfig(config)
        self.tracks_manager_config = TracksManagerConfig(config)
        self.face_id_config = FaceIdConfig(config)

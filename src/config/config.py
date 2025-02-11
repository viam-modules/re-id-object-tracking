from viam.proto.app.robot import ServiceConfig

from src.config.attribute import (
    BoolAttribute,
    FloatAttribute,
    IntAttribute,
    StringAttribute,
)


class TrackerConfig:
    def __init__(self, config: "ServiceConfig"):
        self.re_id_threshold = FloatAttribute(
            field_name="re_id_threshold",
            config=config,
            min_value=0,
            default_value=0.3,
        )

        self.min_track_persistence = IntAttribute(
            field_name="min_track_persistence",
            config=config,
            min_value=0,
            default_value=10,
        )
        self.lambda_value = FloatAttribute(
            field_name="lambda_value",
            config=config,
            min_value=0,
            max_value=1,
            default_value=0.95,
        )
        self.max_age_track = IntAttribute(
            field_name="max_age_track",
            config=config,
            min_value=0,
            max_value=100000,
            default_value=1000,
        )
        self.min_distance_threshold = FloatAttribute(
            field_name="min_distance_threshold",
            config=config,
            min_value=0,
            max_value=1,
            default_value=0.3,
        )
        self.feature_distance_metric = StringAttribute(
            field_name="feature_distance_metric",
            config=config,
            default_value="cosine",
            allowlist=["cosine", "euclidean"],
        )

        self.max_frequency = FloatAttribute(
            field_name="max_frequency_hz",
            config=config,
            default_value=10,
            min_value=0.1,
            max_value=100,
        )

        self.cooldown_period = FloatAttribute(
            field_name="cooldown_period_s",
            config=config,
            default_value=5,
            min_value=0,
        )

        self.start_fresh = BoolAttribute(
            field_name="start_fresh",
            config=config,
            default_value=False,
        )

        self.save_to_db = BoolAttribute(
            field_name="save_to_db",
            config=config,
            default_value=True,
        )

        self._start_background_loop = BoolAttribute(
            field_name="_start_background_loop", config=config, default_value=True
        )

        self.path_to_known_persons = StringAttribute(
            field_name="path_to_known_persons",
            config=config,
            default_value=None,
        )


class DetectorConfig:
    def __init__(self, config: "ServiceConfig"):
        self.model_name = StringAttribute(
            field_name="detector_model_name",
            config=config,
            default_value="fasterrcnn_mobilenet_v3_large_320_fpn",
            allowlist=["fasterrcnn_mobilenet_v3_large_320_fpn"],
        )
        self.threshold = FloatAttribute(
            field_name="detection_threshold",
            config=config,
            min_value=0.0,
            max_value=1.0,
            default_value=0.8,
        )
        self.device = StringAttribute(
            field_name="detector_device",
            config=config,
            default_value="cpu",
            allowlist=["cpu", "cuda"],
        )

        # TODO: add usage for torchvision
        self.max_results = IntAttribute(
            field_name="detection_max_detection_results",
            config=config,
            default_value=5,
            min_value=1,
        )


class FaceIdConfig:
    def __init__(self, config: ServiceConfig):
        self.path_to_known_faces = StringAttribute(
            field_name="path_to_known_faces",
            config=config,
            default_value=None,
        )

        self.device = StringAttribute(
            field_name="face_detector_device",
            config=config,
            default_value="cpu",
            allowlist=["cpu", "cuda"],
        )

        self.detector = StringAttribute(
            field_name="face_detector_model",
            config=config,
            default_value="ultraface_version-RFB-320-int8",
        )

        self.detector_threshold = FloatAttribute(
            field_name="face_detection_threshold",
            config=config,
            min_value=0.0,
            max_value=1.0,
            default_value=0.9,
        )

        self.feature_extractor = StringAttribute(
            field_name="face_feature_extractor_model",
            config=config,
            default_value="facenet",
        )

        self.cosine_id_threshold = FloatAttribute(
            field_name="cosine_id_threshold",
            config=config,
            min_value=0.0,
            max_value=1.0,
            default_value=0.3,
        )

        self.euclidean_id_threshold = FloatAttribute(
            field_name="euclidean_id_threshold",
            config=config,
            min_value=0.0,
            max_value=1.0,
            default_value=0.9,
        )


class FeatureEncoderConfig:
    def __init__(self, config: ServiceConfig):
        self.feature_extractor_name = StringAttribute(
            field_name="feature_extractor_model",
            config=config,
            default_value="osnet_ain_x1_0",
            allowlist=["osnet_x0_25", "osnet_ain_x1_0"],
        )

        self.device = StringAttribute(
            field_name="feature_encoder_device",
            config=config,
            default_value="cuda",
            allowlist=["cpu", "cuda"],
        )


class TracksManagerConfig:
    def __init__(self, config: "ServiceConfig"):
        self.path_to_db = StringAttribute(
            field_name="path_to_database", config=config, required=True
        )

        self.save_period = IntAttribute(
            field_name="save_period",
            config=config,
            default_value=20,
        )


class ReIDObjetcTrackerConfig:
    def __init__(self, config: ServiceConfig):
        self.config = config

        self.tracker_config = TrackerConfig(config)
        self.detector_config = DetectorConfig(config)
        self.encoder_config = FeatureEncoderConfig(config)
        self.tracks_manager_config = TracksManagerConfig(config)
        self.face_id_config = FaceIdConfig(config)

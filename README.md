# VIAM RE-ID OBJECT TRACKER

This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a model of vision service for tracking object using ReID.


![example](img/output.gif)

## Getting started

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/registry/modular-resources/#configuration) and select the `viam:vision:re-id-object-tracker` model from the [`re-id-object-tracker` module](https://app.viam.com/module/re-id-object-tracker).
This module implements the following methods of the [vision service API](https://docs.viam.com/services/vision/#api):

- `GetDetections()`: returns the bounding boxes with the unique id as label and the object detection confidence as confidence.
- `GetClassifications()`: returns the label `new_object_detected` for an image when a new object enters the scene.
- `CaptureAllFromCamera()`: returns the next image and detections or classifications all together, given a camera name.


## Installation
*in progress*

## Configure your `re-id-object-tracker` vision service

> [!NOTE]
> Before configuring your vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

Navigate to the [**CONFIGURE** tab](https://docs.viam.com/configure/) of your [machine](https://docs.viam.com/fleet/machines/) in the [Viam app](https://app.viam.com/).
[Add vision / re-id-object-tracker to your machine](https://docs.viam.com/configure/#components).

### Attributes description

The following attributes are required to configure your `re-id-object-tracker` module. If the
database file does not exist yet, it will be created.

```json
{
  "camera_name": "camera-1",
  "path_to_database": "/path/to/database.db",
  #optional
  "crop_region": {
    "x2_rel": 1,
    "y2_rel": 1,
    "x1_rel": 0.5,
    "y1_rel": 0.5
  }
}
```

## `DoCommand()`

In addition to the [vision service API](https://docs.viam.com/services/vision/#api), the `re-id-object-tracking` module supports some model-specific commands that allow you to add, delete, relabel and list people.
You can invoke these commands by passing appropriately keyed JSON documents to the [`DoCommand()`](/appendix/apis/services/vision/#docommand) method using one of [Viam's SDKs](https://docs.viam.com/sdks/).


### `list_current`

The `list_current` doCommand is used to get all the information of the currently detected tracks.

Input:

```json
"list_current": true
```

returns:
```json
{
  "list_current": {
    track_id": {
      "manual_label": str,
      "face_id_label": str,
      "face_id_conf": float,
      "re_id_label": str,
      "re_id_conf": float
    }
  }
}
```

### `relabel()`

The object tracker generates by default a unique ID string in the format  `"<category>_N_YYYYMMDD_HHMMSS"`. Given this unique id, the user can add a label to track (attached to the `manual_label` field in the output of `list_current`).

```json
"relabel": {"person_N_20241126_190034": "Known Person"}
```

returns:
```json
{
  "relabel": {
    "person_N_20241126_190034": "success: changed label to 'Known Person' "
  }
}
```


### `recompute_embeddings`

Recomputes embeddings.
```json
"recompute_embeddings": true
```

## Supplementaries

### General attributes

| Name                      | Type   | Inclusion | Default       | Description                                                                                    |
| ------------------------- | ------ | --------- | ------------- | ---------------------------------------------------------------------------------------------- |
| `camera_name` | string | **Required** |         | Camera name to be used as input for tracking.               |
| `path_to_database` | string | **Required** |         | Path to the database where tracking information is stored.                                       |
| `lambda_value`            | float  | Optional  | `0.95`        | The lambda value is meant to adjust the contribution of the re-id and the IoU matchings. The distance between two tracks equals: λ * feature_dist + (1 - λ) * (1 - IoU_score). |
| `max_age_track`           | int    | Optional  | `1e3`         | Maximum age (in frames) for a track to be considered active. Ranges from 0 to 1e5.             |
| `min_distance_threshold`  | float  | Optional  | `0.3`         | Minimum distance threshold for considering two tracks as distinct. Values range from 0 to 5. |
| `feature_distance_metric` | string | Optional  | `'cosine'` | Metric used for calculating feature distance. Options include `cosine` and `euclidean`. Refer to [torch-re-id model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html) to select the metric that matches your model.|
| `cooldown_period_s`       | float  | Optional  | `5`           | Duration for which the trigger is on.`new_object_detected`.                                          |
| `re_id_threshold`         | float  | Optional  | `0.3`         | Threshold for determining whether two persons match based on body features similarity.    |
| `min_track_persistence`   | int    | Optional  | `4`          | Minimum number of frames a track candidate must persist before beinfg promoted to a track.                         |
| `max_frequency_hz`           | float  | Optional  | `10`          | Frequency at which the tracking steps are performed. |
| `save_to_db`              | bool   | Optional  | `True`        | Indicates whether tracks should be saved to the database.                       |
| `save_period`      | int    | Optional     | `20`    | Interval (in number of tracking steps) when tracks are saved to the database.               |
| `start_fresh`             | bool   | Optional  | `False`       | Whether or not to load the tracks from the database at `reconfigure()`.                             |
| `path_to_known_persons`   | string | Optional  | `None`        | Path to the database containing pictures of entire persons. If the directory does not exist it will be created at `reconfigure()`. Refer [example directory tree](#example-of-directory-tree) to see how to add pictures of known persons and associate labels with the persons.            |
| `crop_region`             | dict   | Optional  | `None`        | Defines a region of the image to crop for processing. Must include four float values between 0 and 1: `x1_rel`, `y1_rel`, `x2_rel`, `y2_rel` representing the relative coordinates of the crop region. |



### Person detector attributes

| Name                              | Type   | Inclusion    | Default                                 | Description                                                                                                     |
| --------------------------------- | ------ | ------------ | --------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `detector_model_name`             | string | Optional| `'fasterrcnn_mobilenet_v3_large_320_fpn'` | Name of the model used for detection. Only option at the moment. Options include `'fasterrcnn_mobilenet_v3_large_320_fpn'` (low resolution)  and `'fasterrcnn_mobilenet_v3_large_fpn'` (high resolution)  |
| `detection_threshold`             | float  | Optional     | `0.95`                                   | Confidence threshold for detecting objects, with values ranging from 0.0 to 1.0.                                |
| `detector_device`                 | string | Optional     | `'cpu'`                                 | Device on which the detection model will run. Options are `cpu` and `gpu`.                                      |
| `_enable_debug_tools` | bool | Optional | `False` | When enabled, saves images containing person detections to a debug directory. |
| `_path_to_debug_directory` | string | Optional | `None` | Directory path where debug images will be saved. Required if `_enable_debug_tools` is True. |
| `_max_size_debug_directory` | int | Optional | `200` | Maximum number of debug images to store in the debug directory. |


### Feature encoder attributes

| Name                      | Type   | Inclusion | Default        | Description                                                                                                       |
| ------------------------- | ------ | --------- | -------------- | ----------------------------------------------------------------------------------------------------------------- |
| `feature_extractor_model` | string | Optional  | `'osnet_ain_x1_0'` | Name of the model used for feature extraction. Only option at the moment.|
| `feature_encoder_device`  | string | Optional  | `'cuda'`        | Device on which the feature encoder will run. Options are `cpu` and `cuda`.                                        |

### Face re-identification attributes

| Name                      | Type   | Inclusion | Default                             | Description                                                                                                       |
| ------------------------- | ------ | --------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `path_to_known_faces`     | string | Optional  | `None`                              | Path to a file or database containing images or embeddings of known faces. If the directory does not exist it will be created at `reconfigure()`. Refer [example directory tree](#example-of-directory-tree) to see how to add pictures of known faces and associate labels with the faces. |
| `face_detector_device`    | string | Optional  | `'cpu'`                             | Device on which the face detector will run. Options are `cpu` and `cuda`.                                         |
| `face_detector_model`     | string | Optional  | `'ultraface_version-RFB-320-int8'` | Name of the model used for face detection. Only option at the moment.                                                                        |
| `face_detection_threshold`| float  | Optional  | `0.9`                               | Confidence threshold for detecting faces, with values ranging from 0.0 to 1.0.                                   |
| `face_feature_extractor_model` | string | Optional | `'facenet'` | Model used for extracting features from detected faces for identification. Only option at the moment.                                      |
| `cosine_id_threshold`     | float  | Optional  | `0.3`                               | Threshold for determining face identity matches using cosine similarity. Both cosine and euclidean distances should be under threshold for faces to be considered as match.                                          |
| `euclidean_id_threshold`  | float  | Optional  | `0.9`                               | Threshold for determining face identity matches using Euclidean distance.                                             |

### TracksManagerConfig

| Name               | Type   | Inclusion    | Default | Description                                                                                     |
| ------------------ | ------ | ------------ | ------- | ------------------------------------------------------------------------------------------------ |
| `path_to_database` | string | **Required** |         | Path to the database where tracking information is stored.                                       |
| `save_period`      | int    | Optional     | `20`    | Interval (in number of tracking steps) when tracks are saved to the database.               |


## Makefile (arm-jetson machines only)

This project includes a `Makefile` script to automate the PyInstaller build process. PyInstaller is used to create standalone executables from the Python module scripts.

####  `make setup`

1. install system dependencies (cuDNN and cuSPARSELt)
2. create venv environment (under `./build/.venv`)
3. Get python packages wheel files - Torch, ONNXRuntime-GPU, Torchvision (built from source)

Cleaned with `make clean` (this also deletes pyinstaller build directory)

#### `make pyinstaller`
This command builds the module executable using PyInstaller.

This creates the PyInstaller executable under `./build/pyinstaller_dist`.

The `first_run.sh` script is in `bin/first_run.sh`

To upload to viam registry:
```bash
viam login
tar -czvf archive.tar.gz meta.json main first_run.sh  #needs to be on the same level
viam module upload --version 0.0.0-rc0 --platform linux/arm64 --tags 'jetpack:6' archive.tar.gz
```

Cleaned with `make clean-pyinstaller`


### Example of directory tree

In the example below, all persons (or faces) detected in any pictures within the directory `French_Team` will have an embedding associated with the label `French_Team`. The supported image formats for known faces are PNG and JPEG.
```
path
└── to
    └── known_faces
        └── Zinedine_Zidane
        │   └── zz_1.png
        │   └── zz_2.jpeg
        │   └── zz_3.jpeg
        │
        └── Jacques_Chirac
        │   └── jacques_1.jpeg
        │
        └── French_Team
        |   └── ribery.jpeg
        |   └── vieira.png
        |   └── thuram.jpeg
        |   └── group_picture.jpeg
        │
        └── Italian_Team
            └── another_group_picture.png
```

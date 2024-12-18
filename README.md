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

The following attributes are required to configure your `re-id-object-tracker` module:

```json
{
  "camera_name": "camera-1",
  "path_to_database": "/path/to/database.db" #the file doesn't need to exists
}
```

## `DoCommand()`

In addition to the [vision service API](https://docs.viam.com/services/vision/#api), the `re-id-object-tracking` module supports some model-specific commands that allow you to add, delete, relabel and list people.
You can invoke these commands by passing appropriately keyed JSON documents to the [`DoCommand()`](/appendix/apis/services/vision/#docommand) method using one of [Viam's SDKs](https://docs.viam.com/sdks/).


### `list_current`
The `list_current` doCommand is used to get all the informations of the currently detected tracks.

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
The object tracker generate by default a unique ID string in the format  `"<category>_N_YYYYMMDD_HHMMSS"`. Given this unique id, the user can add a label to track (attached to the `manual_label` field in the output of `list_current`).

```json
"relabel": {"person_N_20241126_190034": "Known Person"}
```

returns:
```json
{
  "relabel": {
    "person_N_20241126_190034": "success: changed label to 'Known Person' "
  }
```


### `recompute_embeddings`
Recomputes embeddings.
```json
"recompute_embeddings": true
```

## Supplementaries

### TrackerConfig

| Name                      | Type   | Inclusion | Default       | Description                                                                                    |
| ------------------------- | ------ | --------- | ------------- | ---------------------------------------------------------------------------------------------- |
| `lambda_value`            | float  | Optional  | `0.95`        | The lambda value is meant to adjust the contribution of the re-id matching and the IoU. The distance between two tracks equals: λ * feature_dist + (1 - λ) * (1 - IoU_score). |
| `max_age_track`           | int    | Optional  | `1e3`         | Maximum age (in frames) for a track to be considered active. Ranges from 0 to 1e5.             |
| `min_distance_threshold`  | float  | Optional  | `0.3`         | Minimum distance threshold for considering two features as distinct. Values range from 0 to 5. |
| `feature_distance_metric` | string | Optional  | `'euclidean'` | Metric used for calculating feature distance. Options include `cosine` and `euclidean`.        |
| `cooldown_period_s`       | float  | Optional  | `5`           | Duration for which the trigger is on.`new_object_detected`.                                          |
| `start_fresh`             | bool   | Optional  | `False`       | Whether or not to load the tracks from the database.                                     |
| `re_id_threshold`         | float  | Optional  | `0.3`         | Threshold for determining whether two persons match based on body features similarity.    |
| `min_track_persistence`   | int    | Optional  | `10`          | Minimum number of frames a track must persist to be considered valid.                         |
| `max_frequency`           | float  | Optional  | `10`          | Frequency at which the tracking steps are performed. |
| `save_to_db`              | bool   | Optional  | `True`        | Indicates whether tracking information should be saved to the database.                       |
| `path_to_known_persons`   | string | Optional  | `None`        | Path to the database containing pictures of entire persons.                |

### DetectorConfig

| Name                              | Type   | Inclusion    | Default                                 | Description                                                                                                     |
| --------------------------------- | ------ | ------------ | --------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `detector_model_name`             | string | Optional| `'fasterrcnn_mobilenet_v3_large_320_fpn'` | Name of the model used for detection. Only option at the moment. |
| `detection_threshold`             | float  | Optional     | `0.6`                                   | Confidence threshold for detecting objects, with values ranging from 0.0 to 1.0.                                |
| `detector_device`                 | string | Optional     | `'cpu'`                                 | Device on which the detection model will run. Options are `cpu` and `gpu`.                                      |


### FeatureEncoderConfig

| Name                      | Type   | Inclusion | Default        | Description                                                                                                       |
| ------------------------- | ------ | --------- | -------------- | ----------------------------------------------------------------------------------------------------------------- |
| `feature_extractor_model` | string | Optional  | `'osnet_ain_x1_0'` | Name of the model used for feature extraction. Only option at the moment.|
| `feature_encoder_device`  | string | Optional  | `'cuda'`        | Device on which the feature encoder will run. Options are `cpu` and `cuda`.                                        |

### FaceIdConfig

| Name                      | Type   | Inclusion | Default                             | Description                                                                                                       |
| ------------------------- | ------ | --------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `path_to_known_faces`     | string | Optional  | `None`                              | Path to a file or database containing images or embeddings of known faces.                                       |
| `face_detector_device`    | string | Optional  | `'cpu'`                             | Device on which the face detector will run. Options are `cpu` and `cuda`.                                         |
| `face_detector_model`     | string | Optional  | `'ultraface_version-RFB-320-int8'` | Name of the model used for face detection.  Only option at the moment.                                                                        |
| `face_detection_threshold`| float  | Optional  | `0.9`                               | Confidence threshold for detecting faces, with values ranging from 0.0 to 1.0.                                   |
| `face_feature_extractor_model` | string | Optional | `'facenet'` | Model used for extracting features from detected faces for identification.  Only option at the moment.                                      |
| `cosine_id_threshold`     | float  | Optional  | `0.3`                               | Threshold for determining face identity matches using cosine similarity.                                              |
| `euclidean_id_threshold`  | float  | Optional  | `0.9`                               | Threshold for determining face identity matches using Euclidean distance.                                             |

### TracksManagerConfig

| Name               | Type   | Inclusion    | Default | Description                                                                                     |
| ------------------ | ------ | ------------ | ------- | ------------------------------------------------------------------------------------------------ |
| `path_to_database` | string | **Required** |         | Path to the database where tracking information is stored.                                       |
| `save_period`      | int    | Optional     | `20`    | Interval (in number of tracking steps) when tracks are saved to the database.               |


## PyInstaller Build Process

This project includes a `Makefile` and a `build_installer.sh` script to automate the PyInstaller build process. PyInstaller is used to create standalone executables from the Python module scripts.

### Available Commands

#### 1. `pyinstaller`
This command builds the project using PyInstaller and custom build paths.

##### How it works:
- It runs the `build_installer.sh` script, passing the custom `WORKPATH` and `DISTPATH` environment variables that define where the build artifacts and the final executable will be placed.
- If no values for `WORKPATH` and `DISTPATH` are specified, it uses the default paths defined in the `Makefile`:
  - `WORKPATH`: `./pyinstaller_build`
  - `DISTPATH`: `./pyinstaller_dist`

##### Usage:

To build the project with the default paths:

```bash
make pyinstaller
```

To build the project with custom paths:
```bash
make pyinstaller WORKPATH=/path/to/custom_build DISTPATH=/path/to/custom_dist
```


#### 2. `clean-pyinstaller`

This command removes the directories used by PyInstaller to store the build artifacts and distribution files. It's helpful for cleaning up the project and starting fresh.

##### Usage:

To build the project with the default paths:

```bash
make clean-pyinstaller
```

To clean up with custom paths:
```bash
make clean-pyinstaller WORKPATH=/path/to/custom_build DISTPATH=/path/to/custom_dist
```


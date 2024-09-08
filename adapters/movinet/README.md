# MoViNet Adapter

This repo is a model integration
between [TensorFlow MoviNet](https://www.tensorflow.org/hub/tutorials/movinet) model
and [Dataloop](https://dataloop.ai/).
MoViNet (Mobile Video Networks) is a family of *video classification* models designed for efficient real-time video
processing. It is optimized for mobile and edge devices, making it ideal for real-time applications such as video
surveillance, sports analysis, and more.

## Modes of Operation

The mode of the model can be controlled from the model configuration.

### 1. **Base Mode**

In base mode, MoViNet processes an entire video clip as a single unit. This mode is suitable for tasks where you have
access to the entire video in advance, such as offline video processing or batch inference.

- **Pros**: Suitable for high-quality, offline processing of full videos.
- **Cons**: Requires the entire video to be available beforehand, making it less suitable for real-time applications.

### 2. **Stream Mode**

Stream mode enables real-time video processing by processing a video frame-by-frame (or in small segments) as the video
is received. It is ideal for scenarios where video data is streamed continuously, such as live video feeds.

- **Pros**: Suitable for real-time applications.
- **Cons**: More challenging to optimize due to the need for efficient frame-by-frame processing.

## Installation

To install the package and create the MoviNet model adapter, you will need
a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and
a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the
Dataloop platform. The dataset should
have [directories](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-directory)
containing its training and validation subsets.

## Cloning

For instruction how to clone the pretrained model for prediction
click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#predicting)

## Training and Fine-tuning

For fine tuning on a custom dataset,
click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset)

### Editing the configuration

To edit configurations via the platform, go to the MoviNet model page in the Model Management and edit the json
file displayed there or, via the SDK, by editing the model configuration.
Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more
information.

The basic configurations included are:

* ```hub_version```: The version of the model from the TensorFlow Hub (default: ```3```)
* ```model_id```: The specific model identifier used (default: ```a0```)
* ```model_mode```: Specifies the mode of the model, either base or stream (default: ```base```)
* ```batch_size```:  batch size (default: ```2```)
* ```num_epochs```: The total number of epochs to train the model (default ```8```)
* ```n_frames```: The number of frames to process in each clip (default ```13```)
* ```top_k```: The number of top predictions to consider during prediction (
  default ```3```)





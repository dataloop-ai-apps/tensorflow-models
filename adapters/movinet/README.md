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

**Important Notes:**
1. Currently, only training in base mode is supported. Stream mode can be used for inference but not for training. If you need to fine-tune a model, make sure to use base mode.
2. Make sure to adjust podtype to your needs (larger models versions may require larger podtypes).

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

**Note:** Currently, training is only supported for models in base mode. Stream mode models cannot be trained using this adapter.

### Editing the configuration

To edit configurations via the platform, go to the MoviNet model page in the Model Management and edit the json
file displayed there or, via the SDK, by editing the model configuration.
Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more
information.

The basic configurations included are:

* ```hub_version```: The version of the model from the TensorFlow Hub (default: ```3```)
* ```model_id```: The specific model identifier used (default: ```a0```). MoViNet offers several model sizes:
  * ```a0```: Smallest model size
  * ```a1-a4```: Medium model sizes
  * ```a5```: Largest model size
  
  In general, larger models provide better accuracy but require more computational resources.
* ```model_mode```: Specifies the mode of the model, either base or stream (default: ```base```)
* ```imgz```: The dimensions to resize video frames to (width=imgz, height=imgz) during preprocessing for inference (default: ```224```)
* ```batch_size```: Batch size for training and inference (default: ```2```)
* ```num_epochs```: The total number of epochs to train the model (default: ```8```)
* ```n_frames```: The number of frames to process in each clip (default: ```13```)
* ```top_k```: The number of top predictions to consider during prediction (default: ```3```)
* ```learning_rate```: Learning rate used during model training (default: ```0.001```)

### Top-k Predictions
By default, the model returns the top 3 highest probability predictions for each video. You can configure this by modifying the `top_k` value in the model configuration. For instance, if you set `top_k` to 1, the model will return the most likely activity classification for each video.

## License

The MoViNet model from TensorFlow is distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). When using this model adapter, you must comply with the terms of this license.





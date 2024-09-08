import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import dtlpy as dl
import tempfile
import logging
import cv2
import os

logger = logging.getLogger('movinet-adapter')


@dl.Package.decorators.module(name='model-adapter',
                              description='MoviNet model adapter',
                              init_inputs={'model_entity': dl.Model})
class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model
            This function is called by load_from_model (download to local and then loads)

        :param local_path: `str` directory path in local FileSystem
        """
        weights_filename = self.configuration.get('weights_filename', None)
        self.model_id = self.configuration.get('model_id', 'a0')  # [a0-a5]
        self.model_mode = self.configuration.get('model_mode', 'base')  # ['base','stream']
        hub_version = self.configuration.get('hub_version', 3)
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

        if weights_filename:
            model_path = os.path.join(local_path, weights_filename)
            if os.path.isfile(model_path):
                with tf.device(self.device):
                    self.model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
                logger.info(f"Loaded models weights : {model_path}")
            else:
                raise dl.exceptions.NotFound(
                    f"weights_filename {weights_filename} not found! Make sure to upload the model's weight file as an artifact to the model!")
        else:
            logger.warning("Loading pre-trained weights from hub")
            with tf.device(self.device):
                self.model = self.load_movinet_from_hub(model_id=self.model_id,
                                                        model_mode=self.model_mode,
                                                        hub_version=hub_version)

    def train(self, data_path, output_path, **kwargs):
        logger.warning("Training not implemented yet")

    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of images

            Virtual method - need to implement

        :param batch: `np.ndarray`
        :return: `list[dl.AnnotationCollection]` each collection is per each image / item in the batch
        """
        top_k = self.configuration.get("top_k", 3)
        labels = self.model_entity.labels

        batch_annotations = list()
        for video in batch:

            ################
            # Prepare item #
            ################

            _, suffix = os.path.splitext(video.name)
            buffer = video.download(save_locally=False)

            # Save the buffer to a temporary file for reading with opencv
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
                temp_video.write(buffer.read())
                temp_video_path = temp_video.name
            frames = []
            cap = cv2.VideoCapture(temp_video_path)
            ret = True
            while ret:
                ret, img = cap.read()  # read one frame from the 'capture' object; img is (H, W, C)
                if ret:
                    frames.append(img)

            cap.release()
            video_frames = np.stack(frames, axis=0)  # dimensions (T, H, W, C)
            # Normalize the video frames - byte data is not directly normalized,
            # for ensure that the video frames are correctly scaled for the model.
            video_frames = video_frames / 255.0

            # Delete the temporary video file
            os.remove(temp_video_path)

            ######################
            ######################

            if self.model_mode == 'stream':
                video_with_batch = np.expand_dims(video_frames,
                                                  axis=0)  # add batch size 1 : (batch, frames, height, width, colors)

                init_states_fn = self.model.layers[-1].resolved_object.signatures['init_states']
                init_states = init_states_fn(tf.shape(video_frames[tf.newaxis]))

                with tf.device(self.device):
                    logits, states = self.model.predict({**init_states, 'image': video_with_batch}, verbose=0)
                    logits = logits[0]
                    probs = tf.nn.softmax(logits, axis=-1)

            # Base Mode - video as input, and returns the probabilities averaged over the frames.
            elif self.model_mode == 'base':
                with tf.device(self.device):
                    outputs = self.model.predict(video_frames[tf.newaxis])[0]
                    probs = tf.nn.softmax(outputs)

            else:
                raise dl.exceptions.PlatformException(
                    f"{self.model_mode} is not recognize for MoviNet - can use only 'stream' or 'base'")

            predictions = self.get_top_k(probs=probs, label_map=labels, k=top_k)
            collection = dl.AnnotationCollection(item=video)  # creating collection by item for fps

            for object_id, prediction in enumerate(predictions):
                collection.add(annotation_definition=dl.Classification(label=prediction[0]),
                               object_id=object_id,
                               frame_num=0,
                               end_frame_num=video_frames.shape[0] - 1,
                               model_info={'name': self.model_entity.name,
                                           'confidence': prediction[1],
                                           'model_id': self.model_entity.id})
            batch_annotations.append(collection)

        return batch_annotations

    def prepare_item_func(self, item: dl.Item):
        if 'video' not in item.mimetype:
            raise ValueError('This Model receives only videos as an input')
        return item

    def convert_from_dtlpy(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param data_path: `str` local File System directory path where
                           we already downloaded the data from dataloop platform
        :return:
        """

    @staticmethod
    def load_movinet_from_hub(model_id, model_mode, hub_version=3):
        """Loads a MoViNet model from TF Hub."""

        hub_url = f'https://tfhub.dev/tensorflow/movinet/{model_id}/{model_mode}/kinetics-600/classification/{hub_version}'

        encoder = hub.KerasLayer(hub_url, trainable=True)  # Loads the model as a Keras layer

        inputs = tf.keras.layers.Input(  # Defines the input tensor, flexible input size, suitable for video frames
            shape=[None, None, None, 3],
            dtype=tf.float32)

        if model_mode == 'base':
            inputs = dict(image=inputs)  # Only the video input is required.
        else:
            # Define the state inputs, which is a dict that maps state names to tensors.
            #  This setup is necessary for models that require internal state tracking.
            init_states_fn = encoder.resolved_object.signatures['init_states']
            state_shapes = {
                name: ([s if s > 0 else None for s in state.shape], state.dtype)
                for name, state in init_states_fn(tf.constant([0, 0, 0, 0, 3])).items()
            }
            states_input = {
                name: tf.keras.Input(shape[1:], dtype=dtype, name=name)
                for name, (shape, dtype) in state_shapes.items()
            }

            # The inputs to the model are the states and the video
            inputs = {**states_input,
                      'image': inputs}  # each state name maps to a corresponding input tensor for Keras.

        # Output shape: [batch_size, 600] - 600 is the number of classes in the Kinetics-600 dataset
        outputs = encoder(inputs)

        model = tf.keras.Model(inputs, outputs)  # Creates a Keras model ready for training or inference.

        # Builds the model by input shape, which  allocate resources and optimize the model structure for the given input dimensions.
        model.build([1, 1, 1, 1, 3])
        return model

    @staticmethod
    def get_top_k(probs, label_map, k=5):
        """Outputs the top k model labels and probabilities on the given video."""
        top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
        top_labels = tf.gather(label_map, top_predictions, axis=-1)
        top_labels = [label.decode('utf8') for label in top_labels.numpy()]
        top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
        return tuple(zip(top_labels, top_probs))

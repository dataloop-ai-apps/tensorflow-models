from official.projects.movinet.tools import export_saved_model
from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.modeling import movinet
from frame_generator import FrameGenerator
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import dtlpy as dl
import tempfile
import logging
import cv2
import os

logger = logging.getLogger('segmentation-models-adapter')


@dl.Package.decorators.module(name='model-adapter',
                              description='Base Model Adapter for Segmentation Models',
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
            logger.warning("Loading pre-trained weights from hub")
            with tf.device(self.device):
                self.model = self.load_movinet_from_hub(model_id=self.model_id,
                                                        model_mode=self.model_mode,
                                                        num_classes=len(self.model_entity.labels),
                                                        hub_version=hub_version)

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally
            the function is called in save_to_model which first save locally and then uploads to model entity

        :param local_path: `str` directory path in local FileSystem
        """
        # SAVES IN .h5 SAVING FORMAT
        model_filename = kwargs.get('weights_filename', 'best.h5')
        self.model.save(os.path.join(local_path, model_filename), save_format='h5')
        self.configuration['weights_filename'] = model_filename

    def train(self, data_path, output_path, **kwargs):
        batch_size = self.configuration.get("batch_size", 2)
        num_frames = self.configuration.get("num_frames", 13)
        num_epochs = self.configuration.get("num_epochs", 8)

        # train_subset = self.model_entity.metadata['system']['subsets'].get('train') # TODO: get folder name
        # validation_subset = self.model_entity.metadata['system']['subsets'].get('validation')

        train_subset = 'train'
        validation_subset = 'val'

        output_signature = (tf.TensorSpec(shape=(num_frames, 224, 224, 3), dtype=tf.float32),
                            tf.TensorSpec(shape=(), dtype=tf.int16))

        train_ds = tf.data.Dataset.from_generator(FrameGenerator(local_path=data_path,
                                                                 subset=train_subset,
                                                                 n_frames=num_frames,
                                                                 labels=self.model_entity.labels,
                                                                 label_to_id_map=self.model_entity.label_to_id_map,
                                                                 training=True),

                                                  output_signature=output_signature)
        train_ds = train_ds.batch(batch_size)

        validation_ds = tf.data.Dataset.from_generator(FrameGenerator(local_path=data_path,
                                                                      subset=validation_subset,
                                                                      n_frames=num_frames,
                                                                      labels=self.model_entity.labels,
                                                                      label_to_id_map=self.model_entity.label_to_id_map,
                                                                      training=True),

                                                       output_signature=output_signature)
        validation_ds = validation_ds.batch(batch_size)

        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'], run_eagerly=True)
        tf.config.run_functions_eagerly(True)

        checkpoint_path = os.path.join(data_path, "output", "best_weights.h5")
        # checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            save_best_only=True,  # Save the best model based on the metric (e.g., validation accuracy)
            monitor='val_accuracy',  # Monitor validation accuracy to save the best model
            mode='max',
            verbose=1,
        )

        with tf.device(self.device):
            self.model.fit(train_ds,
                           validation_data=validation_ds,
                           epochs=num_epochs,
                           validation_freq=1,
                           verbose=1,
                           callbacks=[checkpoint_callback])

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

            ######################
            #### Prepare item ####
            ######################

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
    def load_movinet(model_id, batch_size, num_frames, resolution=172, channels=3, num_classes=600):
        use_positional_encoding = model_id in {'a3', 'a4', 'a5'}
        backbone = movinet.Movinet(
            model_id=model_id,
            causal=True,
            conv_type='2plus1d',
            se_type='2plus3d',
            activation='hard_swish',
            gating_activation='hard_sigmoid',
            use_positional_encoding=use_positional_encoding,
            use_external_states=False,
        )

        model = movinet_model.MovinetClassifier(
            backbone,
            num_classes=num_classes,
            output_states=True)

        # Create your example input here.
        # Refer to the paper for recommended input shapes.
        inputs = tf.ones([batch_size, num_frames, resolution, resolution, channels])

        # [Optional] Build the model and load a pretrained checkpoint.
        model.build(inputs.shape)

        return model

    @staticmethod
    def load_movinet_from_hub(model_id, model_mode, num_classes, hub_version=3):
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

        # Replace the output layer to match the number of classes in your dataset
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(outputs)

        model = tf.keras.Model(inputs, outputs)  # Creates a Keras model ready for training or inference.

        # Builds the model by input shape, which  allocate resources and optimize the model structure for the given input dimensions.
        model.build([1, 1, 1, 1, 3])
        # TODO USE INPUTS SIZES BY THE PAPER RECOMMENDED
        return model

    @staticmethod
    def get_top_k(probs, label_map, k=5):
        """Outputs the top k model labels and probabilities on the given video."""
        top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
        top_labels = tf.gather(label_map, top_predictions, axis=-1)
        top_labels = [label.decode('utf8') for label in top_labels.numpy()]
        top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
        return tuple(zip(top_labels, top_probs))


if __name__ == '__main__':
    dl.setenv('rc')
    project = dl.projects.get(project_name="segmentation-models")
    dataset = project.datasets.get(dataset_name="videos")
    model_pretrained = dl.models.get(model_id="66b8b0423c93e233cae207d2")
    model_to_train = dl.models.get(model_id="66b8b0423c93e233cae207d2")
    item1 = dataset.items.get(item_id="66a10d8d9c7ba4c975330331")
    item2 = dataset.items.get(item_id="66b8c1f6b38f787e1feed1fe")

    # # Predict with pre-trained weights
    # adapter = ModelAdapter(model)
    # adapter.predict_items([item1])
    #
    # # Train
    # adapter = ModelAdapter(model_to_train)
    # adapter.train_model(model_to_train)

    # Predict with trained weights
    adapter = ModelAdapter(model_to_train)
    adapter.predict_items([item2])

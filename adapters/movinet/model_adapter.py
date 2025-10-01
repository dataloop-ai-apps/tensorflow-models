from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.modeling import movinet
from adapters.movinet.frame_generator import FrameGenerator
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import dtlpy as dl
import logging
import cv2
import os
import json

logger = logging.getLogger("movinet-model-adapter")


class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        weights_filename = self.configuration.get("weights_filename", None)
        self.model_id = self.configuration.get("model_id", "a0")  # [a0-a5]
        # Validate model_id
        valid_model_ids = ["a0", "a1", "a2", "a3", "a4", "a5"]
        if self.model_id not in valid_model_ids:
            raise ValueError(f"Invalid model_id: {self.model_id}. Must be one of: {', '.join(valid_model_ids)}")

        self.model_mode = self.configuration.get("model_mode", "base")  # ['base','stream']
        hub_version = self.configuration.get("hub_version", 3)
        self.device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        logger.info(f"Device: {self.device}")

        if weights_filename is not None:
            model_path = os.path.join(local_path, weights_filename)
            if os.path.isfile(model_path):
                with tf.device(self.device):
                    backbone = movinet.Movinet(model_id=self.model_id)
                    self.model = movinet_model.MovinetClassifier(
                        backbone=backbone, num_classes=len(self.model_entity.id_to_label_map)
                    )

                    self.model.load_weights(model_path)
                logger.info(f"Loaded models weights : {model_path}")
            else:
                raise dl.exceptions.NotFound(
                    f"weights_filename {weights_filename} not found! Make sure to upload the model's weight file as an artifact to the model!"
                )
        else:
            logger.info("Loading pre-trained weights from hub")
            with tf.device(self.device):
                self.model = self.load_movinet_from_hub(
                    model_id=self.model_id, model_mode=self.model_mode, hub_version=hub_version
                )

    def save(self, local_path, **kwargs):
        # SAVES IN .h5 SAVING FORMAT
        model_filename = "best.h5"
        self.model.save_weights(os.path.join(local_path, model_filename))
        self.configuration["weights_filename"] = model_filename

    def log_metrics_to_dataloop(self, epoch, logs):
        """Log training metrics to Dataloop after each epoch."""
        if logs is None:
            logs = {}
        
        # Update current epoch
        self.current_epoch = epoch + 1
        
        # Define fallback values for non-finite metrics
        NaN_dict = {
            'loss': 0.0,
            'accuracy': 0.0,
            'val_loss': 0.0,
            'val_accuracy': 0.0
        }
        
        # Create a mapping of metric names to figure names for consistency
        metric_mapping = {
            'loss': 'Training Loss',
            'accuracy': 'Training Accuracy', 
            'val_loss': 'Validation Loss',
            'val_accuracy': 'Validation Accuracy'
        }
        
        # Prepare metric samples for Dataloop
        samples = []
        
        for metric_name, value in logs.items():
            if metric_name not in metric_mapping:
                continue
                
            figure_name = metric_mapping[metric_name]
            
            if not np.isfinite(value):
                filters = dl.Filters(resource=dl.FiltersResource.METRICS)
                filters.add(field='modelId', values=self.model_entity.id)
                filters.add(field='figure', values=figure_name)
                filters.add(field='data.x', values=self.current_epoch - 1)
                items = self.model_entity.metrics.list(filters=filters)

                if items.items_count > 0:
                    value = items.items[0].y
                else:
                    value = NaN_dict.get(metric_name, 0)
                logger.warning(f'Value is not finite. For figure {figure_name} and legend metrics using value {value}')
            
            samples.append(dl.PlotSample(figure=figure_name, legend='metrics', x=self.current_epoch, y=value))
        
        # Log metrics to Dataloop
        if samples:
            try:
                self.model_entity.metrics.create(samples=samples, dataset_id=self.model_entity.dataset_id)
                logger.info(f"Logged {len(samples)} metrics for epoch {self.current_epoch}")
            except Exception as e:
                logger.warning(f"Failed to log metrics to Dataloop: {e}")

    def train(self, data_path, output_path, **kwargs):
        if self.model_mode != "base":
            raise ValueError("Only base mode training is supported for MoViNet.")

        self.current_epoch = 0
        
        batch_size = self.configuration.get("batch_size", 2)
        num_frames = self.configuration.get("num_frames", 13)
        num_epochs = self.configuration.get("num_epochs", 8)
        resolution = self.configuration.get("resolution", 172)
        learning_rate = self.configuration.get("learning_rate", 0.001)

        # Load MoviNet without pretrained weights
        tf.keras.backend.clear_session()

        backbone = movinet.Movinet(model_id=self.model_id)
        backbone.trainable = True
        self.model = movinet_model.MovinetClassifier(
            backbone=backbone, num_classes=len(self.model_entity.id_to_label_map)
        )
        self.model.build([batch_size, num_frames, resolution, resolution, 3])

        output_signature = (
            tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int16),
        )

        train_ds = tf.data.Dataset.from_generator(
            FrameGenerator(
                local_path=os.path.join(os.getcwd(), data_path),
                subset="train",
                n_frames=num_frames,
                labels=self.model_entity.labels,
                label_to_id_map=self.model_entity.label_to_id_map,
                model=self.model,
                model_mode=self.model_mode,
                batch_size=batch_size,
                input_size=(resolution, resolution),
                training=True,
            ),
            output_signature=output_signature,
        )
        train_ds = train_ds.batch(batch_size)

        validation_ds = tf.data.Dataset.from_generator(
            FrameGenerator(
                local_path=os.path.join(os.getcwd(), data_path),
                subset="validation",
                n_frames=num_frames,
                labels=self.model_entity.labels,
                label_to_id_map=self.model_entity.label_to_id_map,
                model=self.model,
                model_mode=self.model_mode,
                batch_size=batch_size,
                input_size=(resolution, resolution),
                training=False,
            ),
            output_signature=output_signature,
        )
        validation_ds = validation_ds.batch(batch_size)

        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(loss=loss_obj, optimizer=optimizer, metrics=["accuracy"])
        # tf.config.run_functions_eagerly(True)

        checkpoint_path = os.path.join(data_path, "output", "best_weights.h5")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            save_best_only=True,  # Save the best model based on the metric (e.g., validation accuracy)
            monitor="val_accuracy",  # Monitor validation accuracy to save the best model
            mode="max",
            verbose=1,
        )

        with tf.device(self.device):
            history = self.model.fit(
                train_ds,
                validation_data=validation_ds,
                epochs=num_epochs,
                validation_freq=1,
                verbose=1,
                callbacks=[checkpoint_callback],
            )
            
            # Log metrics for each epoch after training
            for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                history.history.get('loss', []),
                history.history.get('accuracy', []),
                history.history.get('val_loss', []),
                history.history.get('val_accuracy', [])
            )):
                logs = {
                    'loss': loss,
                    'accuracy': acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }
                self.log_metrics_to_dataloop(epoch, logs)

    def predict(self, batch, **kwargs):
        top_k = self.configuration.get("top_k", 3)
        labels = self.model_entity.labels
        imgz = self.configuration.get("imgz", 224)

        batch_annotations = list()
        for video in batch:

            ################
            # Prepare item #
            ################
            os.makedirs(os.path.join(os.getcwd(), "temp_videos"), exist_ok=True)
            file_path = video.download(local_path=os.path.join(os.getcwd(), "temp_videos"))
            
            # Get video dimensions and target size
            cap = cv2.VideoCapture(file_path)
            target_size = self.configuration.get("imgz", 224)
            frames = []
            
            ret = True
            while ret:
                ret, img = cap.read()
                if ret:
                    # Resize frame immediately to save memory
                    resized_frame = cv2.resize(img, (target_size, target_size))
                    # Convert to float32 immediately
                    normalized_frame = resized_frame.astype(np.float32) / 255.0
                    frames.append(normalized_frame)

            cap.release()
            video_frames = np.stack(frames, axis=0)  # dimensions (T, target_size, target_size, 3)
            logger.info(f"Processed video shape: {video_frames.shape}")
            # Delete the temporary video file
            os.remove(file_path)
            logger.info(f"Removed {file_path}")
            ######################
            ######################

            if self.model_mode == "stream":
                logger.info("Predicting - Mode Stream")
                video_with_batch = np.expand_dims(
                    video_frames, axis=0
                )  # add batch size 1 : (batch, frames, height, width, colors)

                init_states_fn = self.model.layers[-1].resolved_object.signatures["init_states"]
                init_states = init_states_fn(tf.shape(video_frames[tf.newaxis]))

                with tf.device(self.device):
                    logger.info(f"Device: {self.device}")
                    logits, states = self.model.predict({**init_states, "image": video_with_batch}, verbose=0)
                    logits = logits[0]  # concatenating all the logits
                    probs = tf.nn.softmax(logits, axis=-1)  # estimating probabilities

            # Base Mode - video as input, and returns the probabilities averaged over the frames.
            elif self.model_mode == "base":
                logger.info("Predicting - Mode Base")
                with tf.device(self.device):
                    logger.info(f"Device: {self.device}")
                    resized_frames = np.stack(
                        [cv2.resize(f, (imgz, imgz)) for f in video_frames]
                    )  # resize to (224, 224) - safer and faster
                    video_tensor = tf.convert_to_tensor(resized_frames.astype("float32"))
                    outputs = self.model.predict(video_tensor[tf.newaxis])[0]
                    probs = tf.nn.softmax(outputs)

            else:
                raise dl.exceptions.PlatformException(
                    f"{self.model_mode} is not recognize for MoviNet - can use only 'stream' or 'base'"
                )

            predictions = self.get_top_k(probs=probs, label_map=labels, k=top_k)
            collection = dl.AnnotationCollection(item=video)  # creating collection by item for fps

            for object_id, prediction in enumerate(predictions):
                collection.add(
                    annotation_definition=dl.Classification(label=prediction[0]),
                    object_id=object_id,
                    frame_num=0,
                    end_frame_num=video_frames.shape[0] - 1,
                    model_info={
                        "name": self.model_entity.name,
                        "confidence": prediction[1],
                        "model_id": self.model_entity.id,
                    },
                )
            batch_annotations.append(collection)

        return batch_annotations

    def prepare_item_func(self, item: dl.Item):
        if "video" not in item.mimetype:
            raise ValueError("This Model receives only videos as an input")
        return item

    def convert_from_dtlpy(self, data_path, **kwargs):
        """Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param data_path: `str` local File System directory path where
                           we already downloaded the data from dataloop platform
        :return:
        """

    @staticmethod
    def load_movinet_from_hub(model_id, model_mode, hub_version=3):
        """Loads a MoViNet model from TF Hub."""

        hub_url = (
            f"https://tfhub.dev/tensorflow/movinet/{model_id}/{model_mode}/kinetics-600/classification/{hub_version}"
        )

        encoder = hub.KerasLayer(hub_url, trainable=True)  # Loads the model as a Keras layer

        inputs = tf.keras.layers.Input(  # Defines the input tensor, flexible input size, suitable for video frames
            shape=[None, None, None, 3], dtype=tf.float32
        )

        if model_mode == "base":
            inputs = dict(image=inputs)  # Only the video input is required.
        else:
            # Define the state inputs, which is a dict that maps state names to tensors.
            #  This setup is necessary for models that require internal state tracking.
            init_states_fn = encoder.resolved_object.signatures["init_states"]
            state_shapes = {
                name: ([s if s > 0 else None for s in state.shape], state.dtype)
                for name, state in init_states_fn(tf.constant([0, 0, 0, 0, 3])).items()
            }
            states_input = {
                name: tf.keras.Input(shape[1:], dtype=dtype, name=name) for name, (shape, dtype) in state_shapes.items()
            }

            # The inputs to the model are the states and the video
            inputs = {
                **states_input,
                "image": inputs,
            }  # each state name maps to a corresponding input tensor for Keras.

        # Output shape: [batch_size, 600] - 600 is the number of classes in the Kinetics-600 dataset
        outputs = encoder(inputs)

        model = tf.keras.Model(inputs, outputs)  # Creates a Keras model ready for training or inference.

        # Builds the model by input shape, which  allocate resources and optimize the model structure for the given input dimensions.
        model.build([1, 1, 1, 1, 3])
        return model

    @staticmethod
    def get_top_k(probs, label_map, k=5):
        """Outputs the top k model labels and probabilities on the given video."""
        top_predictions = tf.argsort(probs, axis=-1, direction="DESCENDING")[:k]
        top_labels = tf.gather(label_map, top_predictions, axis=-1)
        top_labels = [label.decode("utf8") for label in top_labels.numpy()]
        top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
        return tuple(zip(top_labels, top_probs))


if __name__ == "__main__":
    # Training test
    model_entity = dl.models.get(model_id="")
    model_entity.dataset_id = ""
    model_entity.metadata['system'] = {}
    # If items are tagged with train and validation, use the following:
    model_entity.metadata['system']['subsets'] = {
        'train': dl.Filters(field='metadata.system.tags.train', values=True).prepare(),
        'validation': dl.Filters(field='metadata.system.tags.validation', values=True).prepare()
    }
    # If items are not tagged with train and validation, use the following:
    # model_entity.metadata['system']['subsets'] = {
    #     'train': json.dumps(dl.Filters(field='dir', values='/train').prepare()),
    #     'validation': json.dumps(dl.Filters(field='dir', values='/validation').prepare()),
    # }
    model_entity.status = 'pre-trained'
    model_entity.update(True)
    adapter = ModelAdapter(model_entity=model_entity)
    adapter.train_model(model=model_entity)

    # # Prediction test
    # Uncomment and fill in item_id to test prediction
    # model_entity = dl.models.get(model_id="")
    # item = dl.items.get(item_id="") 
    # adapter = ModelAdapter(model_entity=model_entity)
    # adapter.predict_items(items=[item])
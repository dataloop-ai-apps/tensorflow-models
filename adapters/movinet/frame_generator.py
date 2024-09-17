import tensorflow as tf
from glob import glob
import numpy as np
import random
import json
import cv2
import os


def format_frames(frame, output_size):
    """
      Pad and resize an image from a video.

      Args:
        frame: Image that needs to resized and padded.
        output_size: Pixel size of the output frame image.

      Return:
        Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


def frames_from_video_file(video_path, n_frames, input_size=(224, 224), frame_step=15):
    """
      Creates frames from each video file present for each category.

      Args:
        video_path: File path to the video.
        n_frames: Number of frames to be created per video file.
        output_size: Pixel size of the output frame image.

      Return:
        An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, input_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, input_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result


class FrameGenerator:
    def __init__(self, local_path, subset, n_frames, labels, label_to_id_map, model, model_mode, batch_size,
                 input_size=(224, 224),
                 training=False):
        """ Returns a set of frames with their associated label.

          Args:
            path: Video file paths.
            n_frames: Number of frames.
            training: Boolean to determine if training dataset is being created.
        """
        self.path = local_path
        self.subset = subset
        self.n_frames = n_frames
        self.training = training
        self.class_names = labels
        self.class_ids_for_name = label_to_id_map
        self.model_mode = model_mode
        self.batch_size = batch_size
        self.model = model
        self.input_size = input_size

    def get_files_and_class_names(self):
        # Video local paths
        # TODO: HARDCODED VIDEO SUFFIX ?
        # TODO: MP4, .WEBM , ? OPTIONS
        video_paths = glob(os.path.join(self.path, self.subset, 'items', '**', '**/*.webm'), recursive=True)
        video_paths += glob(os.path.join(self.path, self.subset, 'items', '**', '*.webm'), recursive=True)

        # Annotation local paths
        json_files = glob(os.path.join(self.path, self.subset, 'json', '**', '**/*.json'), recursive=True)
        json_files += glob(os.path.join(self.path, self.subset, 'json', '**', '*.json'), recursive=True)

        classes = list()
        # Find matching file names
        for video_path in video_paths:
            # to both Windows and Linux
            path_parts = video_path.split(os.sep)
            # Replace 'items' with 'jsons'
            path_parts = ['json' if part == 'items' else part for part in path_parts]
            path_with_jsons = os.sep.join(path_parts)
            json_path = os.path.splitext(path_with_jsons)[0] + '.json'

            f = open(json_path)
            json_file = json.load(f)
            annotations = json_file.get('annotations')
            classes.append(annotations[0].get('label'))  # TODO MORE THAN 1 LABEL?

        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()
        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(video_path=path, n_frames=self.n_frames, input_size=self.input_size)
            label = self.class_ids_for_name[name]  # Encode labels
            # yield video_frames, label

            if self.model_mode == 'stream':  # TODO: REMOVE
                # Generate initial states for streaming mode
                init_states_fn = self.model.layers[-1].resolved_object.signatures['init_states']
                init_states = init_states_fn(tf.shape(video_frames[tf.newaxis]))

                states = self.model.get_initial_states(batch_size=self.batch_size)
                yield video_frames, states, label
            else:
                yield video_frames, label

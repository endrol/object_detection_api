import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import argparse
import os
from utils.preprocessing_data import resize_and_pad_image
import tensorflow_datasets as tfds
from models.decode_detection import DecodePredictions
from utils.retinanet_loss import RetinaNetLoss
from data_utils.get_dataset import get_dataset
import json
import cv2
from datetime import datetime
from draw_rectangle import draw_rectangle
'''inference model based on save_model or save_weights
use opencv to draw the images and bboxes
'''


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize inferenece results")
    parser.add_argument(
        "-m",
        "--model",
        help="path to the saved model",
        type=str,
        required=True
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        help="path to the test dataset",
        type=str,
        required=True
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        help="path to the output directory",
        type=str,
        required=True
    )

    parser.add_argument(
        '--config',
        help="path to the config file",
        type=str,
        default="pallet.json"
    )

    parser.add_argument(
        "--low_thresh",
        help="Minimum score of the detected boxes to show",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--high_thresh",
        help="heigh threshold that be considered good detection",
        type=float,
        default=0.6,
    )

    args = parser.parse_args()
    return args

def load_config(filename=None):
    if filename is None:
        return {"classes": {"1": "default"}, "colors": {"default": [0, 0, 0]}}
    else:
        with open(filename, "r") as f:
            return json.load(f)

def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


# def visualize_detections(
#     index, image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[1, 0, 1]
# ):
#     """Visualize Detections"""
#     image = np.array(image, dtype=np.uint8)
#     plt.figure(figsize=figsize)
#     plt.axis("off")
#     plt.imshow(image)
#     ax = plt.gca()
#     for box, _cls, score in zip(boxes, classes, scores):
#         text = "{}: {:.2f}".format(_cls, score)
#         x1, y1, x2, y2 = box
#         w, h = x2 - x1, y2 - y1
#         patch = plt.Rectangle(
#             [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
#         )
#         ax.add_patch(patch)
#         ax.text(
#             x1,
#             y1,
#             text,
#             bbox={"facecolor": color, "alpha": 0.4},
#             clip_box=ax.clipbox,
#             clip_on=True,
#         )
#     plt.savefig('/workspace/data/retinanet_save/'+index+"test_save.png")
#     plt.show()
#     return ax


def main():
    args = parse_args()

    inference_model = keras.models.load_model(args.model, compile=False)

    imgs = [
        img_file for img_file in os.listdir(args.input_dir)
        if os.path.splitext(img_file)[1] in [".png", ".jpg"]]
    imgs.sort()
    for img_file in imgs:
        img_path = os.path.join(args.input_dir, img_file)
        img = cv2.imread(img_path)
        cols, rows = img.shape[1], img.shape[0]
        img, _ = prepare_image(img)
        detections = inference_model.predict(img)
        num_detections = int(detections[3][0])

        for i in range(num_detections):
            score = detections[1][0][i]
            if score < args.low_thresh:
                continue
            bbox = detections[2][0][i]
            x1, y1, x2, y2 = bbox
            class_name


    for i, sample in enumerate(val_dataset.take(100)):

        image = tf.cast(sample["image"], dtype=tf.float32)
        input_image, ratio = prepare_image(image)
        # detection is returned by tf.image.combined_non_max_suppression
        detections = inference_model.predict(input_image)
        num_detections = detections[3][0]
        class_names = [class_pallet[int(label_class)] for label_class in detections[2][0][:num_detections]]

        visualize_detections(str(i),
            image,
            detections[0][0][:num_detections] / ratio,
            class_names,
            detections[1][0][:num_detections],
        )


if __name__ == '__main__':
    main()

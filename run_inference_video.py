import pdb
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

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


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
        help="path to the test dataset, bag-image-folder",
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
    image, _ = resize_and_pad_image(image)
    # image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0)


def main():
    args = parse_args()

    inference_model = keras.models.load_model(args.model, compile=False)
    config_data = load_config(args.config)
    classes = config_data["classes"]
    imgs = [
        img_file for img_file in os.listdir(args.input_dir)
        if os.path.splitext(img_file)[1] in [".png", ".jpg"]]
    imgs.sort()

    timestamp = datetime.now()
    folder_timestamp = timestamp.strftime("%Y_%m_%d_%H_%M")
    video_out_pth = args.output_dir + "/" + folder_timestamp + ".avi"
    video_out = cv2.VideoWriter(
        video_out_pth, cv2.VideoWriter_fourcc(*"DIVX"), 24, (300, 300)
    )

    for i, img_file in enumerate(imgs):
        img_path = os.path.join(args.input_dir, img_file)
        img = cv2.imread(img_path)
        # import pdb; pdb.set_trace()
        cols, rows = img.shape[1], img.shape[0]
        img_t = prepare_image(img)
        detections = inference_model.predict(img_t)
        # import pdb; pdb.set_trace()
        num_detections = int(detections[3][0])

        for i in range(num_detections):
            score = detections[1][0][i]
            if score < args.low_thresh:
                continue
            bbox = detections[0][0][i]
            x1, y1, x2, y2 = bbox
            class_name = classes[str(int(detections[2][0][i]))]
            img = draw_rectangle(img, (x1, y1), (x2, y2), class_name["color"], sides=False)

        # write image to video
        # img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite(video_out_pth+str(i)+".jpg", img)

        video_out.write(img)
    video_out.release()


if __name__ == '__main__':
    main()

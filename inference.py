import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.preprocessing_data import resize_and_pad_image
import tensorflow_datasets as tfds
from models.decode_detection import DecodePredictions
from utils.retinanet_loss import RetinaNetLoss
'''inference model based on save_model or save_weights
'''


class_voc = ['background',
             'aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat',
             'chair', 'cow', 'diningtable', 'dog',
             'horse', 'motorbike', 'person', 'pottedplant',
             'sheep', 'sofa', 'train', 'tvmonitor']


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize inferenece results")
    parser.add_argument(
        "-m",
        "--model",
        help="path to the saved model",
        type=str,
        required=True
    )
    args = parser.parse_args()
    return args


def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


def visualize_detections(
    index, image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[1, 0, 1]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.savefig('/workspace/data/save_retinanet/'+index+"test_save.png")
    plt.show()
    return ax


# test_dataset = get_dataset(filepath='/workspace/data/cats_dogs/train/pets.tfrecord',
#                            batch_size=batch_size,
#                            is_training=False,
#                            for_test=True)
# val_dataset = tfds.load("voc/2007", split="validation", data_dir="data")
# int2str = dataset_info.features["objects"]["label"].int2str


def main():
    args = parse_args()

    inference_model = keras.models.load_model(args.model, compile=False)
    val_dataset = tfds.load("voc/2007", split="validation", data_dir="data")

    for i, sample in enumerate(val_dataset.take(100)):

        image = tf.cast(sample["image"], dtype=tf.float32)
        input_image, ratio = prepare_image(image)
        # detection is returned by tf.image.combined_non_max_suppression
        detections = inference_model.predict(input_image)
        num_detections = detections[3][0]
        class_names = [class_voc[int(label_class)] for label_class in detections[2][0][:num_detections]]

        visualize_detections(str(i),
            image,
            detections[0][0][:num_detections] / ratio,
            class_names,
            detections[1][0][:num_detections],
        )


if __name__ == '__main__':
    main()

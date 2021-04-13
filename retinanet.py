import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pdb
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from utils.label_encoder import LabelEncoder
from data_utils.get_dataset import get_dataset
from models.backbones import get_backbone
from models.retinanet import RetinaNet
from models.decode_detection import DecodePredictions
from utils.retinanet_loss import RetinaNetLoss
from utils.preprocessing_data import preprocess_data, resize_and_pad_image
from datetime import datetime


model_dir = "retinanet/"

num_classes = 2
batch_size = 2

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    )
]

'''load dataset both customized and official
'''
# train_dataset = get_dataset(filepath='/workspace/data/cats_dogs/train/pets.tfrecord',
#                             batch_size=batch_size,
#                             is_training=True)
# val_dataset = get_dataset(filepath='/workspace/data/cats_dogs/valid/pets.tfrecord',
#                           batch_size=batch_size,
#                           is_training=False)
label_encoder = LabelEncoder()
(train_dataset, val_dataset), dataset_info = tfds.load(
    "voc/2007", split=["train", "validation"], with_info=True, data_dir="data"
)

autotune = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
train_dataset = train_dataset.shuffle(8 * batch_size)
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
train_dataset = train_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(
    batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)


# Uncomment the following lines, when training on full dataset
# train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
# val_steps_per_epoch = \
#     dataset_info.splits["validation"].num_examples // batch_size

# train_steps = 4 * 100000
# epochs = train_steps // train_steps_per_epoch

epochs = 500

# Running 100 training and 50 validation steps,
# remove `.take` when training on the full dataset

# model.fit(
#     train_dataset,
#     validation_data=val_dataset.take(50),
#     epochs=epochs,
#     callbacks=callbacks_list,
#     verbose=1,
# )

# Change this to `model_dir` when not using the downloaded weights
weights_dir = model_dir

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)


# TODO tf.keras.applications.resnet.preprocess_input(image),
# can we customize this?
def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[1, 0, 1]
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
    timenow = datetime.now()
    timestamp = timenow.strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(timestamp+"test_save.png")
    plt.show()
    return ax


# test_dataset = get_dataset(filepath='/workspace/data/cats_dogs/train/pets.tfrecord',
#                            batch_size=batch_size,
#                            is_training=False,
#                            for_test=True)
val_dataset = tfds.load("voc/2007", split="validation", data_dir="data")
int2str = dataset_info.features["objects"]["label"].int2str

for sample in val_dataset.take(2):
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )

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
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, TerminateOnNaN, ReduceLROnPlateau
from models.backbones import get_backbone
from models.retinanet import RetinaNet
from models.decode_detection import DecodePredictions
from utils.retinanet_loss import RetinaNetLoss
from utils.preprocessing_data import preprocess_data, resize_and_pad_image
from datetime import datetime




num_classes = 20
batch_size = 8

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

time_stamp = datetime.now().strftime("%m_%d_%H_%M")
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='/workspace/object_detection_api/tensorboard_save/'+time_stamp+'/train/'+"weights" + "_epoch_{epoch}",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch',
        verbose=1,
        period=2
    )
]

def image_test(
    image, figsize=(7, 7)
):
    """Visualize Detections"""
    # image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    plt.savefig("test_save.png")
    plt.show()

'''load dataset both customized and official
'''
train_dataset = get_dataset(filepath='/workspace/data/voc_tf/train.record',
                            batch_size=batch_size,
                            is_training=True)
val_dataset = get_dataset(filepath='/workspace/data/voc_tf/val.record',
                          batch_size=batch_size,
                          is_training=False)
# label_encoder = LabelEncoder()
# (train_dataset_tfds, val_dataset_tfds), dataset_info = tfds.load(
#     "voc/2007", split=["train", "validation"], with_info=True, data_dir="data"
# )
# for sample in train_dataset_tfds.take(1):
#     image_test(sample["image"])
#     pdb.set_trace()

# autotune = tf.data.experimental.AUTOTUNE
# train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
# train_dataset = train_dataset.shuffle(8 * batch_size)
# train_dataset = train_dataset.padded_batch(
#     batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
# )
# train_dataset = train_dataset.map(
#     label_encoder.encode_batch, num_parallel_calls=autotune
# )
# train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
# train_dataset = train_dataset.prefetch(autotune)

# val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
# val_dataset = val_dataset.padded_batch(
#     batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
# )
# val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
# val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
# val_dataset = val_dataset.prefetch(autotune)


tensorboard = TensorBoard(log_dir='/workspace/object_detection_api/tensorboard_save/'+time_stamp+'/',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True,
                          update_freq='epoch',
                          profile_batch=2,
                          embeddings_freq=0)
csv_logger = CSVLogger(filename='/workspace/object_detection_api/tensorboard_save/'+time_stamp+'/train/'+'training_log.csv',
                        separator='.',
                        append=True)
terminate_on_nan = TerminateOnNaN()
# reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

callbacks_list.extend([tensorboard, csv_logger,terminate_on_nan])

epochs = 4
start_epoch = 0
weights_dir = ''
if len(weights_dir) > 0:
    start_epoch = int(weights_dir.split('epoch_')[1])
    model.load_weights(weights_dir)
# Change this to `model_dir` when not using the downloaded weights
# /workspace/object_detection_api/retinanet/weights_epoch_17

# latest_checkpoint = tf.train.latest_checkpoint(weights_dir)



# Running 100 training and 50 validation steps,
# remove `.take` when training on the full dataset

model.fit(
    train_dataset,
    validation_data=val_dataset.take(50),
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
    initial_epoch=start_epoch
)

pdb.set_trace()


model.save('/workspace/object_detection_api/tensorboard_save/'+time_stamp+'/frozen/model')

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
val_dataset = tfds.load("voc/2007", split="validation", data_dir="data")
# int2str = dataset_info.features["objects"]["label"].int2str

for i, sample in enumerate(val_dataset.take(100)):
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [class_voc[int(label_class)] for label_class in detections.nmsed_classes[0][:num_detections]]
    # pdb.set_trace()
    visualize_detections(str(i),
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )

import tensorflow as tf
import sys
sys.path.append('..')
from utils.label_encoder import LabelEncoder
from utils.preprocessing_data import preprocess_data
import matplotlib.pyplot as plt
import numpy as np
'''a tf dataset pipeline
'''
# TODO get dataset from images as well

def decode_image(image):
    image = tf.io.decode_jpeg(image, channels=3)
    # if not IS_TRAINING:
    #     image = tf.image.resize(image, IMG_SIZE)
    return image


def read_tfrecord(example):
    tfrecord_format = (
        {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64)
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image/encoded'])
    img_height = tf.dtypes.cast(example['image/height'], tf.float32)
    img_width = tf.dtypes.cast(example['image/width'], tf.float32)
    object_label = example['image/object/class/label']
    object_label = tf.dtypes.cast(object_label, tf.float32)
    object_label = tf.sparse.to_dense(object_label)
    bboxes_xmin = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    bboxes_xmax = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    bboxes_ymin = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    bboxes_ymax = tf.sparse.to_dense(example['image/object/bbox/ymax'])

    sample = {}
    sample['image'] = image
    sample['objects'] = {}
    sample['objects']['bbox'] = tf.stack([bboxes_ymin,bboxes_xmin,bboxes_ymax,bboxes_xmax], axis=-1)
    sample['objects']['label'] = object_label

    return sample


def load_dataset(file):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(file)

    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord)

    return dataset

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

def get_dataset(filepath, batch_size, is_training, for_test=False):
    autotune = tf.data.experimental.AUTOTUNE
    label_encoder = LabelEncoder()
    file = tf.io.gfile.glob(filepath)
    dataset = load_dataset(file)

    # for sample in dataset.take(1):
    #     image_test(sample["image"])
    #     import pdb; pdb.set_trace()


    if for_test:
        return dataset
    # process dataset by map-function
    dataset = dataset.map(preprocess_data, num_parallel_calls=autotune)
    if is_training:
        dataset = dataset.shuffle(8 * batch_size)
    dataset = dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    dataset = dataset.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    )
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.prefetch(autotune)

    # val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
    # val_dataset = val_dataset.padded_batch(
    #     batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    # )
    # val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    # val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    # val_dataset = val_dataset.prefetch(autotune)
    return dataset

def main():
    train_dataset = get_dataset(filepath='/workspace/object_detection_api/data/voc/2007/4.0.0/voc-test.tfrecord-00001-of-00004',
                                batch_size=2)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()

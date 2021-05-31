import pdb
import tensorflow as tf
import sys
sys.path.append('..')
from utils.anchor_generator import AnchorBox
from utils.bbox_util import convert_to_corners


class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )


def _decode_box_predictions(anchor_boxes, box_predictions):
    _box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )
    boxes = box_predictions * _box_variance
    boxes = tf.concat(
        [
            boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
            tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
        ],
        axis=-1,
    )
    boxes_transformed = convert_to_corners(boxes)
    return boxes_transformed


def decode_prediction(confidence_threshold, image_shape, predictions, num_classes):
    _anchor_box = AnchorBox()
    anchor_boxes = _anchor_box.get_anchors(image_shape[0], image_shape[1])

    box_predictions = predictions[:, :, :4]
    cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
    boxes = _decode_box_predictions(anchor_boxes[None, ...], box_predictions)
    boxes = tf.reshape(boxes, [-1, 4])
    scores = tf.reshape(cls_predictions, [-1, num_classes])
    labels = tf.math.argmax(scores, axis=1)
    scores = tf.math.reduce_max(scores, axis=1)
    pdb.set_trace()
    unique_labels, which_one_in_unique_labels = tf.unique(labels)
    chosen_indices = tf.zeros([0])
    for index, unique_label in enumerate(unique_labels):
        indices = tf.reshape(tf.where(which_one_in_unique_labels == index), -1)
        selected_indices, _ = tf.image.non_max_suppression_with_scores(
            tf.gather(boxes, indices),
            tf.gather(scores, indices),
            100,
            0.5,
            0.5
        )
        chosen_indices = tf.concat([chosen_indices, tf.gather(indices, selected_indices)], axis=0)
    big_range_scores = tf.gather(scores, chosen_indices)
    final_indices = tf.gather(chosen_indices, tf.math.top_k(big_range_scores, k=100).indices)
    pdb.set_trace()
    return (tf.gather(boxes, final_indices), tf.gather(scores, final_indices), tf.gather(labels, final_indices), final_indices.shape[0])

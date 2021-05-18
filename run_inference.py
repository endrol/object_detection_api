#!/usr/bin/env python
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import pdb

from tensorflow.python.keras.utils.generic_utils import default

'''run inference based on the pb model like v1-pipeline
frozen_inference_graph.pb
TODO
'''
# /workspace/ssd_mobilenet_v1/data/video_test/0420_test/frozen_inference_graph.pb
# /workspace/object_detection_api/tensorboard_save/04_20_03_18/frozen/model/saved_model.pb


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
        '--output_dir',
        help="output path to the frozen inference graph",
        type=str,
        default="/workspace/data/video_test/0513_test"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # load model
    model = keras.models.load_model(args.model, compile=False)
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    for layer in layers:
        print(layer)
    print('**********')
    print("Frozen model inputs: ", frozen_func.inputs)
    print('**************')
    print("Frozen model outputs: ", frozen_func.outputs)

    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=args.output_dir,
        name="frozen_inference_graph.pb",
        as_text=False)


if __name__ == '__main__':
    main()

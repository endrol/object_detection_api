import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import config
from data_utils.get_dataset import get_dataset
from tensorflow.keras.callbacks import (
    TensorBoard,
    CSVLogger,
    TerminateOnNaN,
    ModelCheckpoint,
)
from models.backbones import get_backbone
from models.retinanet import RetinaNet
from models.decode_detection import DecodePredictions, decode_prediction
from utils.retinanet_loss import RetinaNetLoss
from datetime import datetime
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize inferenece results")
    parser.add_argument(
        "--traindata",
        help="path to the train dataset, tfrecord",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--validdata",
        help="path to the valid dataset. tfrecord",
        type=str,
        required=True,
    )

    parser.add_argument("--n_classes", help="number of classes", type=int, default=10)

    parser.add_argument("--batchsize", help="batch size", type=int, default=8)

    parser.add_argument(
        "--epochs", help="number of training epochs", type=int, default=35
    )

    parser.add_argument(
        "-ckp",
        "--checkpoint",
        help="path to the load checkpoint weight",
        type=str,
        default="",
    )

    args = parser.parse_args()
    return args


def image_test(image, figsize=(7, 7)):
    """Visualize Detections"""
    # image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    plt.savefig("test_save.png")
    plt.show()


def get_callbacks(time_stamp):
    root_dir = os.getcwd()

    tensorboard = TensorBoard(
        log_dir=os.path.join(root_dir, "tensorboard_save/") + time_stamp + "/",
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        update_freq="epoch",
        profile_batch=2,
        embeddings_freq=0,
    )

    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(root_dir, "tensorboard_save/")+time_stamp + "/train/" + "weights" + "_epoch_{epoch}",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        verbose=1,
    )

    csv_logger = CSVLogger(
        filename=os.path.join(root_dir, "tensorboard_save/")
        + time_stamp
        + "/train/"
        + "training_log.csv",
        separator=".",
        append=True,
    )

    terminate_on_nan = TerminateOnNaN()
    return [tensorboard, model_checkpoint, csv_logger, terminate_on_nan]


def get_optimizer():
    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    return optimizer


def main():
    args = parse_args()
    time_stamp = datetime.now().strftime("%m_%d_%H_%M")
    wandb.init(
        project="object_detection_api",
        config={
            "dataset": args.traindata,
            "batchsize": args.batchsize,
            "n_classes": args.n_classes,
            "epochs": args.epochs,
            "checkpoint": args.checkpoint,
            "f_name": time_stamp
        },
        sync_tensorboard=True
        )
    wandb_config = wandb.config

    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(wandb_config.n_classes)
    model = RetinaNet(wandb_config.n_classes, resnet50_backbone)

    model.compile(loss=loss_fn, optimizer=get_optimizer())

    """load dataset both customized and official
    """
    train_dataset = get_dataset(
        filepath=args.traindata, batch_size=wandb_config.batchsize, is_training=True
    )
    val_dataset = get_dataset(
        filepath=args.validdata, batch_size=wandb_config.batchsize, is_training=False
    )

    # load checkpoint if parsed
    start_epoch = 0
    if len(wandb_config.checkpoint) > 0:
        start_epoch = int(wandb_config.checkpoint.split("epoch_")[1])
        model.load_weights(wandb_config.checkpoint)

    # Running 100 training and 50 validation steps,
    # remove `.take` when training on the full dataset

    model.fit(
        train_dataset,
        validation_data=val_dataset.take(50),
        epochs=wandb_config.epochs,
        callbacks=get_callbacks(time_stamp),
        verbose=1,
        initial_epoch=start_epoch,
    )

    model.save(
        os.path.join(os.getcwd(), "tensorboard_save/") + time_stamp + "/frozen/model"
    )
    image = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(image, training=False)
    detections = decode_prediction(confidence_threshold=0.5, image_shape=[224, 224], predictions=predictions, num_classes=10)
    inference_model = tf.keras.Model(inputs=image, outputs=detections)
    inference_model.save(
        os.path.join(os.getcwd(), "tensorboard_save/") + time_stamp + "/frozen/model"
    )


if __name__ == "__main__":
    main()

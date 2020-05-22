from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLOv4,YOLOv3, YOLOv3_tiny, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './data/yolov4.weights', 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    trainset = Dataset('train')
    testset = Dataset('test')
    logdir_train = "./data/log/train"
    logdir_train_epoch = "./data/log/train_epoch"
    logdir_test_epoch = "./data/log/test_epoch"
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    STRIDES         = np.array(cfg.YOLO.STRIDES)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    XYSCALE = cfg.YOLO.XYSCALE
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)

    if FLAGS.tiny:
        feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
    else:
        if FLAGS.model == 'yolov3':
            feature_maps = YOLOv3(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
        elif FLAGS.model == 'yolov4':
            feature_maps = YOLOv4(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)

    model.summary()

    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            if FLAGS.tiny:
                utils.load_weights_tiny(model, FLAGS.weights)
            else:
                if FLAGS.model == 'yolov3':
                    utils.load_weights_v3(model, FLAGS.weights)
                else:
                    utils.load_weights_for_train(model, FLAGS.weights)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)


    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(logdir_train): shutil.rmtree(logdir_train)
    if os.path.exists(logdir_train_epoch): shutil.rmtree(logdir_train_epoch)
    if os.path.exists(logdir_test_epoch): shutil.rmtree(logdir_test_epoch)
    writer_train = tf.summary.create_file_writer(logdir_train)
    writer_train_epoch = tf.summary.create_file_writer(logdir_train_epoch)
    writer_test_epoch = tf.summary.create_file_writer(logdir_test_epoch)

    train_total_loss = []
    train_giou_loss = []
    train_conf_loss = []
    train_prob_loss = []

    test_total_loss = []
    test_giou_loss = []
    test_conf_loss = []
    test_prob_loss = []

    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(3):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))

            train_total_loss.append(total_loss)
            train_giou_loss.append(giou_loss)
            train_conf_loss.append(conf_loss)
            train_prob_loss.append(prob_loss)

            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer_train.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("step/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("step/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("step/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("step/prob_loss", prob_loss, step=global_steps)
            writer_train.flush()
    def test_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(3):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            
            test_total_loss.append(total_loss)
            test_giou_loss.append(giou_loss)
            test_conf_loss.append(conf_loss)
            test_prob_loss.append(prob_loss)

    for epoch in range(first_stage_epochs + second_stage_epochs):
        train_total_loss = []
        train_giou_loss = []
        train_conf_loss = []
        train_prob_loss = []

        test_total_loss = []
        test_giou_loss = []
        test_conf_loss = []
        test_prob_loss = []
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
                    freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
                    freeze = model.get_layer(name)
                    unfreeze_all(freeze)
        for image_data, target in trainset:
            train_step(image_data, target)
        for image_data, target in testset:
            test_step(image_data, target)
        
        # writing summary data
        with writer_train_epoch.as_default():
            tf.summary.scalar("epoch/total_loss", tf.reduce_mean(train_total_loss), step=epoch)
            tf.summary.scalar("epoch/giou_loss", tf.reduce_mean(train_giou_loss), step=epoch)
            tf.summary.scalar("epoch/conf_loss", tf.reduce_mean(train_conf_loss), step=epoch)
            tf.summary.scalar("epoch/prob_loss", tf.reduce_mean(train_prob_loss), step=epoch)
        writer_train_epoch.flush()
        with writer_test_epoch.as_default():
            tf.summary.scalar("epoch/total_loss", tf.reduce_mean(test_total_loss), step=epoch)
            tf.summary.scalar("epoch/giou_loss", tf.reduce_mean(test_giou_loss), step=epoch)
            tf.summary.scalar("epoch/conf_loss", tf.reduce_mean(test_conf_loss), step=epoch)
            tf.summary.scalar("epoch/prob_loss", tf.reduce_mean(test_prob_loss), step=epoch)
        writer_test_epoch.flush()
        model.save_weights(f"./checkpoints/yolov4_{epoch}")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
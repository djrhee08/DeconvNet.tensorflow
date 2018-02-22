from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorlayer as tl # pip install tensorlayer

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import read_DICOMbatch as dicom_batch
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
import time

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(20)
NUM_OF_CLASSES = 2
IMAGE_SIZE = 224


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )

    pool_argmax = []
    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w",)
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current, pool_arg_temp = utils.pool_layer(current)
            pool_argmax.append(pool_arg_temp)
        net[name] = current

    print("length of pool_argmax : ", len(pool_argmax))
    print(pool_argmax[-1].shape,pool_argmax[-2].shape,pool_argmax[-3].shape, pool_argmax[-4].shape, pool_argmax[-5].shape)

    return net, pool_argmax


def inference(image, keep_prob):
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net, pool_argmax = vgg_net(weights, processed_image)
        conv_final_layer = image_net["pool5"]

        fc_6 = utils.conv_layer(x=conv_final_layer, W_shape=[7, 7, 512, 4096], b_shape=4096, name='fc_6', padding='SAME')
        fc_7 = utils.conv_layer(x=fc_6, W_shape=[1, 1, 4096, 4096], b_shape=4096, name='fc_7', padding='SAME')

    with tf.variable_scope("Deconv"):
        deconv_fc_6 = utils.deconv_layer(fc_7, [7, 7, 512, 4096], 512, 'fc6_deconv')

        #unpool_5 = utils.unpool_layer2x2(deconv_fc_6, pool_argmax[-1], tf.shape(image_net["conv5_3"]))
        unpool_5 = utils.unpool_layer2x2_batch(deconv_fc_6, pool_argmax[-1]) # Use unpool_layer2x2_batch if the input image is a batch

        deconv_5_3 = utils.deconv_layer(unpool_5, [3, 3, 512, 512], 512, 'deconv_5_3')
        deconv_5_2 = utils.deconv_layer(deconv_5_3, [3, 3, 512, 512], 512, 'deconv_5_2')
        deconv_5_1 = utils.deconv_layer(deconv_5_2, [3, 3, 512, 512], 512, 'deconv_5_1')

        #unpool_4 = utils.unpool_layer2x2(deconv_5_1, pool_argmax[-2], tf.shape(image_net["conv4_3"]))
        unpool_4 = utils.unpool_layer2x2_batch(deconv_5_1, pool_argmax[-2])

        deconv_4_3 = utils.deconv_layer(unpool_4, [3, 3, 512, 512], 512, 'deconv_4_3')
        deconv_4_2 = utils.deconv_layer(deconv_4_3, [3, 3, 512, 512], 512, 'deconv_4_2')
        deconv_4_1 = utils.deconv_layer(deconv_4_2, [3, 3, 256, 512], 256, 'deconv_4_1')

        #unpool_3 = utils.unpool_layer2x2(deconv_4_1, pool_argmax[-3], tf.shape(image_net["conv3_3"]))
        unpool_3 = utils.unpool_layer2x2_batch(deconv_4_1, pool_argmax[-3])

        deconv_3_3 = utils.deconv_layer(unpool_3, [3, 3, 256, 256], 256, 'deconv_3_3')
        deconv_3_2 = utils.deconv_layer(deconv_3_3, [3, 3, 256, 256], 256, 'deconv_3_2')
        deconv_3_1 = utils.deconv_layer(deconv_3_2, [3, 3, 128, 256], 128, 'deconv_3_1')

        #unpool_2 = utils.unpool_layer2x2(deconv_3_1, pool_argmax[-4], tf.shape(image_net["conv2_2"]))
        unpool_2 = utils.unpool_layer2x2_batch(deconv_3_1, pool_argmax[-4])

        deconv_2_2 = utils.deconv_layer(unpool_2, [3, 3, 128, 128], 128, 'deconv_2_2')
        deconv_2_1 = utils.deconv_layer(deconv_2_2, [3, 3, 64, 128], 64, 'deconv_2_1')

        #unpool_1 = utils.unpool_layer2x2(deconv_2_1, pool_argmax[-5], tf.shape(image_net["conv1_2"]))
        unpool_1 = utils.unpool_layer2x2_batch(deconv_2_1, pool_argmax[-5])


        deconv_1_2 = utils.deconv_layer(unpool_1, [3, 3, 64, 64], 64, 'deconv_1_2')
        deconv_1_1 = utils.deconv_layer(deconv_1_2, [3, 3, 32, 64], 32, 'deconv_1_1')

        score_1 = utils.deconv_layer(deconv_1_1, [1, 1, 21, 32], 21, 'score_1')

        logits = tf.reshape(score_1, (-1, 21))

        prediction = tf.argmax(tf.reshape(tf.nn.softmax(logits), tf.shape(score_1)), dimension=3)
        print(prediction.shape, logits.shape, deconv_1_2.shape, deconv_1_1.shape)

    return prediction, logits


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)

    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")

    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    pred_annotation, logits = inference(image, keep_probability)

    print("pred_annotation, logits shape", pred_annotation.get_shape().as_list(), logits.get_shape().as_list(), annotation.get_shape().as_list())
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(annotation, [-1]), name="entropy")))  # For softmax

    trainable_var = tf.trainable_variables()

    train_op = train(loss, trainable_var)

#    for variable in trainable_var:
#        print(variable)

    #Way to count the number of variables + print variable names
    """
    total_parameters = 0
    for variable in trainable_var:
        # shape is an array of tf.Dimension
        print(variable)
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print("Total # of parameters : ", total_parameters)
    """

    print("Setting up image reader...")
    dir_name='AQA/'
    contour_name='external'

    batch_size = 1
    rotation = True
    rotation_angle = [-10,-5,5,10]
    bitsampling = False
    bitsampling_bit = [4, 8]
    resize_shape = (224, 224)
    dicom_records = dicom_batch.read_DICOM(dir_name=dir_name+'training_set', contour_name=contour_name, resize_shape=resize_shape,
                                           rotation=rotation, rotation_angle=rotation_angle,
                                           bitsampling=bitsampling, bitsampling_bit=bitsampling_bit)
    validation_records = dicom_batch.read_DICOM(dir_name=dir_name+'validation_set', contour_name=contour_name, resize_shape=resize_shape,
                                           rotation=False, rotation_angle=rotation_angle,
                                           bitsampling=False, bitsampling_bit=bitsampling_bit)


    sess = tf.Session() # Need GPU here for max_pool_with_argmax
    #sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) # CPU ONLY

    print("Setting up Saver...")
    saver = tf.train.Saver()
    #summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        start = time.time()
        train_loss_list = []
        validation_loss_list = []
        # for itr in xrange(MAX_ITERATION):
        for itr in xrange(10): # about 18 hours of work
            train_images, train_annotations = dicom_records.next_batch(batch_size=batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if (itr+1) % 5 == 0:
                #train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                train_loss = sess.run(loss, feed_dict=feed_dict)

                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_loss_list.append(train_loss)
                #summary_writer.add_summary(summary_str, itr)

            if (itr+1) % 5 == 0:
                valid_images, valid_annotations = validation_records.next_batch(batch_size=batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                validation_loss_list.append(valid_loss)

            if (itr+1) % 200 == 0:
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr+1)

            end = time.time()
            print("Iteration #", itr+1, ",", np.int32(end - start), "s")

        saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr+1)
        print("train_loss_list : ", train_loss_list)
        print("validation_loss_list : ", validation_loss_list)


    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_records.next_batch(batch_size=batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        print(pred.shape, valid_annotations.shape)
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)
        print(pred.shape, valid_annotations.shape)

        # Save the image for display. Use matplotlib to draw this.
        for itr in range(1):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)


    # Need to add another mode to draw the contour based on image only.

if __name__ == "__main__":
    tf.app.run()
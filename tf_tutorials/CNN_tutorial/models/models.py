#############################################################################
##                                                                         ##
## model.py:       defines models to classify handwritten                  ##
##                 numbers of the MNIST dataset using low level TensorFlow ##
##                 API and the tf.data API.                                ##
##                                                                         ##
#############################################################################

import numpy as np
import os 
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# TODO: 
# 1) modify train so that it saves checkpoints of the graph's vars along with the data
#    that has yet to be trained on so that if training is interrupted, I can continue
#    training the model on the dataset.
#
# 2) dtype and otype can be regular python data types as opposed to tensorflow data types

        

"""
hyperparameters
Training: batch_size, epochs
Layer_1: output_depth, convo_kernal_shape, strides, padding, use_bias, act_func,
         init_func_w, init_func_b, init_func_params_w, init_func_params_b
Layer_2: m_pl_kernal_shape, strides, padding
Layer_3: output_width, use_bias, act_func, init_func_w, init_func_b, init_func_params_w
         init_func_params_b
Layer_o: init_func_w, init_func_b, init_func_params_w, init_func_params_b

"""

class model(object):
    def __init__(self, directory, dtype, dshape, otype, oshape, hyparams):
        self.directory = directory
        self.data_type = dtype
        self.data_shape = dshape
        self.out_type = otype
        self.out_shape = oshape
        self.__hparams = hyparams
        self.tf_graph = self.build_graph()
        with self.tf_graph.as_default():
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph = self.tf_graph)
        self.writer = tf.summary.FileWriter(self.directory)
        self.writer.add_graph(graph = self.sess.graph)
        self.writer.flush()
        checkpoint = tf.train.get_checkpoint_state(self.directory + '\\checkpoints\\')
        if checkpoint is not None:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            
        
    @property
    def hparams(self):
        return self.__placeholder
    @hparams.setter
    def __hparams(self, vals):
        hparam_types = {"batch_size": type(1), "epochs": type(1), "L1_output_depth": type(1),
                        "L1_convo_kernal_shape": type([]), "L1_strides": type([]),
                        "L1_padding": type(""), "L1_use_bias": type(True),
                        "L1_act_func": type(lambda v: v), "L1_init_func_w": type(lambda v:v), 
                        "L1_init_func_b": type(lambda v: v), "L1_init_func_params_w": type({}),
                        "L1_init_func_params_b": type({}), "L2_m_pl_kernal_shape": type([]),
                        "L2_strides": type([]), "L2_padding": type(""), "L3_output_width": type(1),
                        "L3_use_bias": type(True), "L3_act_func": type(lambda v: v),
                        "L3_init_func_w": type(lambda v: v), "L3_init_func_b": type(lambda v: v),
                        "L3_init_func_params_w": type({}), "L3_init_func_params_b": type({}),
                        "Lo_init_func_w": type(lambda v: v), "Lo_init_func_b": type(lambda v: v),
                        "Lo_init_func_params_w": type({}), "Lo_init_func_params_b": type({}),
                        "Lo_use_bias": type(True)}

        if (type(vals) == type({})):
            for key in hparam_types.keys():
                if (key not in vals) or (hparam_types[key] != type(vals[key])):
                    raise ValueError("hyper parameters must contain key: {} with value of type: {}".format(key, hparam_types[key]))
            self.__placeholder = vals
        else:
            raise ValueError("hyparams must be an object of type: dictionary")
    
    # Trains model data_set   
    def train(self, data_set, labels):
        train_op, cost, accuracy, prediction, raw_prediction, merged_summary, init_op, iter_init_op, x_placeholder, y_placeholder = self.tf_graph.get_collection("nodes_to_run")
        with self.sess.as_default():
            self.sess.run(init_op)
            for epoch in range(self.hparams["epochs"]):
                epoch = epoch + 1
                self.sess.run(iter_init_op, feed_dict = {x_placeholder: data_set, y_placeholder: labels})
                batch_num = 1
                while True:
                    try:
                        C, acc, _ = self.sess.run([cost, accuracy, train_op])
                        if batch_num % 10 == 0:
                            sm = self.sess.run(merged_summary)
                            self.writer.add_summary(sm, batch_num)
                            print("Epoch: {}, Batch: {}, Cost: {:.3f}, accuracy: {:.2f}%".format(epoch, batch_num, C, acc))
                        batch_num = batch_num + 1
                    except tf.errors.OutOfRangeError:
                        break

    # Evaluates model on data set = data_set, and labels = labels
    #  ==> {"cost": number, "accuracy": number}
    def evaluate(self, data_set, labels):
        train_op, cost, accuracy, prediction, raw_prediction, merged_summary, init_op, iter_init_op, x_placeholder, y_placeholder = self.tf_graph.get_collection("nodes_to_run")
        with self.sess.as_default():
            self.sess.run(iter_init_op, feed_dict = {x_placeholder: data_set, y_placeholder: labels})
            batch_num = 1
            total_cost = 0
            total_acc = 0
            while True:
                try:
                    C, acc = self.sess.run([cost, accuracy])
                    total_cost = total_cost + C
                    total_acc = total_acc + acc
                    batch_num = batch_num + 1
                except tf.errors.OutOfRangeError:
                    total_cost = total_cost / batch_num
                    total_acc = total_acc / batch_num
                    break
            print("cost over data set: {:.3f}, accuracy over data set: {:.2f}%".format(total_cost, total_acc))
            return {"cost": total_cost, "accuracy": total_acc}
            
    # Saves model to disk        
    def save(self):
        get_global_step_op = tf.train.get_global_step(self.tf_graph)
        global_step = self.sess.run([get_global_step_op])
        self.saver.save(self.sess, self.directory + '\\checkpoints\\tooop', global_step[0])
        # Cant store HParams as a config file because parameter of type function
        # cant be converted to a protocol buffer string or json string.
        # will implement later.
        
        
    """ Functions that are used to build the model's Graph"""
    # Builds a 2D convolution layer
    #  Takes:
    #  name = string that contains no whitespace specifies the name of the layer in the graph
    #  input = tensor of shape [batch_size, input_height, input_width, input_depth]
    #  output_depth = number 1-N, depth of the output tensor
    #  kernal_shape = tensor of shape [kernal_height, kernal_width] specifies the height 
    #                 and width of the 2D matrix that is convolved with a depthwise
    #                 slice of from our 3D data 
    #  strides = tensor of shape [stride_in_batch_size_direction,
    #                             stride_in_height_direction, stride_in_width_direction,
    #                             stride_in_depth_direction]
    #            for the classic mathematic convolution we set strides = [1,1,1,1]
    #  padding = a string of either "SAME" or "VALID". "SAME" means enough 0s are padded 
    #            around the input_heigth and input_width dimensions so that the result of each
    #            2D convolution with one of the 2D kernals is a tensor of the same height and width as 
    #            input. While "VALID" means no padding is applied.
    #  use_bias = boolean. When True we add a bias offset to the results of our convolution operation
    #  act_func = function(tensor of any shape) ==> tensor of shape = size(tensor of any size)
    #  init_func_w/b = function(tensor of any shape *params) ==> tensor of shape = inpupt shape
    #  init_func_params_w/b = dictionary where each key is a the name of a parameter in init_func_w/b
    #                         and the value of a key is the value passed to the parameter
    #  ==> Tensor of size [batch_size, new_height, new_width, output_depth] 
    def build_conv2D_layer(self, name, input, output_depth, kernal_shape, 
                           strides = [1, 1, 1, 1], padding = "SAME", 
                           use_bias = True, act_func = lambda act: act,
                           init_func_w = tf.random_normal,
                           init_func_params_w = {'stddev': 0.05},
                           init_func_b = tf.random_normal,
                           init_func_params_b = {'stddev': 0.05}):
        with tf.name_scope(name):
            input_shape = input.get_shape().as_list()
            kern_shape = kernal_shape[:]
            kern_shape.append(input_shape[3])
            kern_shape.append(output_depth)
            kernals = tf.Variable(init_func_w(kern_shape, **init_func_params_w), name = "kernals")
            output = tf.nn.conv2d(input, kernals, strides, padding, name = "Convolution")
            if use_bias:
                bias = tf.Variable(init_func_b([output_depth], **init_func_params_b), name = "bias")
                output = tf.add(output, bias)
            output = act_func(output)
            return output
            
    # Builds a fully connected neural network layer
    #  Takes:
    #  name = string that contains no whitespace specifies the name of the layer in the graph
    #  input = Tensor of shape [batch_size, features]
    #  width = number 1-N specifies the number of neurons in the layer
    #  act_func = function(tensor of any shape) ==> tensor of shape = input shape
    #  init_func_w/b = function(tensor of any shape *params) ==> tensor of shape = inpupt shape
    #  init_func_params_w/b = dictionary where each key is a the name of a parameter in init_func
    #                         and the value of a key is the value passed to the parameter
    #  ==> Tensor of shape [batch_size, width]
    def build_fc_layer(self, name, input, width, use_bias = True,
                       act_func = lambda act: act,
                       init_func_w = tf.random_normal, 
                       init_func_params_w = {'stddev': 0.05},
                       init_func_b = tf.random_normal,
                       init_func_params_b = {'stddev': 0.5}):
        with tf.name_scope(name):
            input_shape = input.get_shape().as_list()
            weights = tf.Variable(init_func_w([input_shape[1], width], **init_func_params_w), name = "weights")
            activation = tf.matmul(input, weights)
            if use_bias:
                bias = tf.Variable(init_func_b([width], **init_func_params_b), name = "bias")
                activation = tf.add(activation, bias)
            activation = act_func(activation)
            return activation
    
    # Builds the model part of the tensorflow graph
    #  Takes:
    #  name = string with no whitespace, specifies the name of the component in the graph
    #  input = tensor of shape [batch_size, height, width, depth]
    #  ==> tensor of shape [batch_size, 10]
    def build_model(self, name, input):
        with tf.name_scope(name):
            # Building convolution layer
            c_layer_1 = self.build_conv2D_layer("conv_layer_1", input,
                                                output_depth = self.hparams["L1_output_depth"], 
                                                kernal_shape = self.hparams["L1_convo_kernal_shape"],
                                                strides = self.hparams["L1_strides"], padding = self.hparams["L1_padding"],
                                                use_bias = self.hparams["L1_use_bias"], act_func = self.hparams["L1_act_func"],
                                                init_func_w = self.hparams["L1_init_func_w"], init_func_params_w = self.hparams["L1_init_func_params_w"],
                                                init_func_b = self.hparams["L1_init_func_b"], init_func_params_b = self.hparams["L1_init_func_params_b"])
            # Building max pooling layer
            m_pool_layer_1 = tf.nn.max_pool(c_layer_1, ksize = self.hparams["L2_m_pl_kernal_shape"], strides = self.hparams["L2_strides"], padding = self.hparams["L2_padding"])
            # Reshaping m_pool_layer_1 for input to fully connect layers
            layer_shape = m_pool_layer_1.get_shape().as_list()
            new_shape = np.prod(layer_shape[1:])
            m_pool_layer_1 = tf.reshape(m_pool_layer_1, [-1, new_shape])
            # Building fully connected layer
            fc_layer_1 = self.build_fc_layer("fc_layer_1", m_pool_layer_1, width = self.hparams["L3_output_width"],
                                             use_bias = self.hparams["L3_use_bias"], act_func = self.hparams["L3_act_func"],
                                             init_func_w = self.hparams["L3_init_func_w"], init_func_params_w = self.hparams["L3_init_func_params_w"],
                                             init_func_b = self.hparams["L3_init_func_b"], init_func_params_b = self.hparams["L3_init_func_params_b"])
            # Building fully connected output layer
            output_layer = self.build_fc_layer("output_layer", fc_layer_1, width = 10, use_bias = self.hparams["Lo_use_bias"],
                                               init_func_w = self.hparams["Lo_init_func_w"], init_func_params_w = self.hparams["Lo_init_func_params_w"],
                                               init_func_b = self.hparams["Lo_init_func_b"], init_func_params_b = self.hparams["Lo_init_func_params_b"])
            return output_layer
            
        
    # Builds the tensorflow graph object for the model
    #  Takes:
    #  ==> tf.Graph() object 
    def build_graph(self):
        tf_graph = tf.Graph()
        with tf_graph.as_default():
            # Building placeholder nodes to feed data to, and Dataset object 
            #  to preprocess and serve data to the rest of the graph
            with tf.name_scope("input_dataset"):
                x_placeholder = tf.placeholder(self.data_type, [None].extend(self.data_shape))
                y_placeholder = tf.placeholder(self.out_type)
                x_data = tf.data.Dataset.from_tensor_slices(x_placeholder).map(lambda e: tf.reshape(tf.cast(e, tf.float32), [28, 28, 1]))
                y_data = tf.data.Dataset.from_tensor_slices(y_placeholder).map(lambda e: tf.one_hot(e, 10))
                dataset = tf.data.Dataset.zip((x_data, y_data)).shuffle(500).batch(self.hparams["batch_size"])
            # Building initializable iterator object to access the Dataset object
            with tf.name_scope("iterator"):
                iterator = dataset.make_initializable_iterator()
                iter_init_op = iterator.initializer # This operation must be ran to initialize the Dataset and iterator objects
            # Extracting a batch of data from the Dataset object and prepping data 
            images, labels = iterator.get_next()
            # Building training step counter 
            global_step = tf.Variable(0, trainable = False, name = 'global_step')
            # Building model
            logits = self.build_model(name = "model", input = images)
            raw_prediction = tf.nn.softmax(logits)
            # Building cost function 
            with tf.name_scope("cost_function"):
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits))
                tf.summary.scalar("cross_entropy", cost)
            # Building training operations
            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.minimize(cost, global_step = global_step)
            # Building evaluation metrics
            with tf.name_scope("accuracy"):
                prediction = tf.argmax(raw_prediction, 1)
                equality = tf.equal(prediction, tf.argmax(labels, 1))
                accuracy = tf.multiply(tf.reduce_mean(tf.cast(equality, tf.float32)), 100)
                tf.summary.scalar("Accuracy", accuracy)
            # Building variable initialization operation
            init_op = tf.global_variables_initializer()
            # merging all summaries to one summary
            merged_summary = tf.summary.merge_all()
            for node in [train_op, cost, accuracy, prediction, raw_prediction, merged_summary,
                         init_op, iter_init_op, x_placeholder, y_placeholder]:
                tf_graph.add_to_collection("nodes_to_run", node)
        return tf_graph

"""hparams = {"batch_size": 32, "epochs": 1, "L1_output_depth": 5,
              "L1_convo_kernal_shape": [5, 5], "L1_strides": [1, 1, 1, 1],
              "L1_padding": "SAME", "L1_use_bias": True, "L1_act_func": tf.nn.relu,
              "L1_init_func_w": tf.random_normal, "L1_init_func_b": tf.random_normal,
              "L1_init_func_params_w": {'stddev': 0.05}, "L1_init_func_params_b": {'stddev': 0.05},
              "L2_m_pl_kernal_shape": [1, 2, 2, 1], "L2_strides": [1, 2, 2, 1], "L2_padding": "SAME",
              "L3_output_width": 100, "L3_use_bias": True, "L3_act_func": tf.nn.relu,
              "L3_init_func_w": tf.random_normal, "L3_init_func_b": tf.random_normal,
              "L3_init_func_params_w": {'stddev': 0.05}, "L3_init_func_params_b": {'stddev': 0.5},
              "Lo_init_func_w": tf.random_normal, "Lo_init_func_b": tf.random_normal,
              "Lo_init_func_params_w": {'stddev': 0.05}, "Lo_init_func_params_b": {'stddev': 0.5},
              "Lo_use_bias": True}"""
                                      
#test_model = model(".\\test\\", tf.float32, [28, 28], tf.int32, [1], hparams)   
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#test_model.train(x_train, y_train)      

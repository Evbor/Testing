#############################################################################
##                                                                         ##
## model_v1.py:    creates a neural network to classify handwritten        ##
##                 numbers of the MNIST dataset using low level TensorFlow ##
##                 API and the tf.data API.                                ##
##                                                                         ##
#############################################################################

""" hyperparameters: batch_size, use_bias_L1, output_channels_L1, 
                     kernal_shape_L1, x_stride_L1, y_stride_L1, padding_L1,
                     activation_fxn_L1, kernal_shape_L2, x_stride_L2, y_stride_L2
                     padding_L2, width_L3, use_bias_L3, activation_fxn_L3


"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

class model(object):
    def __init__(self, dtype, dshape, otype, oshape, hyparams):
        self.data_type = dtype
        self.data_shape = dshape
        self.out_type = otype
        self.out_shape = oshape
        self.__hparams = hyparams
        self.tf_graph = self.build_graph()
        self.sess = tf.Session(graph = self.tf_graph)
        self.writer = tf.summary.FileWriter("D:\\Programs\\testing\\tf_tutorials\\CNN_tutorial\\models\\testss")
        self.writer.add_graph(graph = self.sess.graph)
        self.writer.flush()
        
    @property
    def hparams(self):
        return self.__placeholder
    @hparams.setter
    def __hparams(self, vals):
        # test if vals is a list of the correct hparams
        self.__placeholder = vals
        
    # Trains model    
    def train(self, data_set, labels):
        train_op, cost, accuracy, prediction, raw_prediction, merged_summary, init_op, iter_init_op, x_placeholder, y_placeholder = self.tf_graph.get_collection("nodes_to_run")
        with self.sess.as_default():
            self.sess.run(init_op)
            for epoch in range(1):
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
    #  init_func = function(tensor of any shape *params) ==> tensor of shape = inpupt shape
    #  init_func_params = dictionary where each key is a the name of a parameter in init_func
    #                     and the value of a key is the value passed to the parameter
    #  ==> Tensor of size [batch_size, new_height, new_width, output_depth] 
    def build_conv2D_layer(self, name, input, output_depth, kernal_shape, 
                           strides = [1, 1, 1, 1], padding = "SAME", 
                           use_bias = True, act_func = lambda act: act,
                           init_func = tf.random_normal,
                           init_func_params = {'stddev': 0.05}):
        with tf.name_scope(name):
            input_shape = input.get_shape().as_list()
            kernal_shape.append(input_shape[3])
            kernal_shape.append(output_depth)
            kernals = tf.Variable(init_func(kernal_shape, **init_func_params), name = "kernals")
            output = tf.nn.conv2d(input, kernals, strides, padding, name = "Convolution")
            if use_bias:
                bias = tf.Variable(init_func([output_depth], **init_func_params), name = "bias")
                output = tf.add(output, bias)
            output = act_func(output)
            return output
            
    # Builds a fully connected neural network layer
    #  Takes:
    #  name = string that contains no whitespace specifies the name of the layer in the graph
    #  input = Tensor of shape [batch_size, features]
    #  width = number 1-N specifies the number of neurons in the layer
    #  act_func = function(tensor of any shape) ==> tensor of shape = input shape
    #  ==> Tensor of shape [batch_size, width]
    def build_fc_layer(self, name, input, width, act_func = lambda act: act):
        with tf.name_scope(name):
            input_shape = input.get_shape().as_list()
            weights = tf.Variable(tf.random_normal([input_shape[1], width], stddev = 0.05), name = "weights")
            bias = tf.Variable(tf.random_normal([width], stddev = 0.5), name = "bias")
            activation = tf.add(tf.matmul(input, weights), bias)
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
            c_layer_1 = self.build_conv2D_layer("conv_layer_1", input, output_depth = 5, 
                                                  kernal_shape = [5, 5],
                                                  act_func = tf.nn.relu)
            # Building max pooling layer
            m_pool_layer_1 = tf.nn.max_pool(c_layer_1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
            # Reshaping m_pool_layer_1 for input to fully connect layers
            layer_shape = m_pool_layer_1.get_shape().as_list()
            new_shape = np.prod(layer_shape[1:])
            m_pool_layer_1 = tf.reshape(m_pool_layer_1, [-1, new_shape])
            # Building fully connected layer
            fc_layer_1 = self.build_fc_layer("fc_layer_1", m_pool_layer_1, 100, act_func = tf.nn.relu)
            # Building fully connected output layer
            output_layer = self.build_fc_layer("output_layer", fc_layer_1, 10)
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
                dataset = tf.data.Dataset.zip((x_data, y_data)).shuffle(500).batch(self.hparams.batch_size)
            # Building initializable iterator object to access the Dataset object
            with tf.name_scope("iterator"):
                iterator = dataset.make_initializable_iterator()
                iter_init_op = iterator.initializer # This operation must be ran to initialize the Dataset and iterator objects
            # Extracting a batch of data from the Dataset object and prepping data 
            images, labels = iterator.get_next()
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
                train_op = optimizer.minimize(cost)
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

hparams = tf.contrib.training.HParams(batch_size = 32)        
test_model = model(tf.float32, [28, 28], tf.int32, [1], hparams)   
(x_train, y_train), (x_test, y_test) = mnist.load_data()
test_model.train(x_train, y_train)      

#############################################################################
##                                                                         ##
## NN_tutorial.py: creates a neural network to classify handwritten        ##
##                 numbers of the MNIST dataset using low level TensorFlow ##
##                 API and the tf.data API.                                ##
##                                                                         ##
#############################################################################

import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Parameters
training_batch_size = 32
epochs = 1

# Importing MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Input to NN is a vector and not a 28x28 matrix
#  therefore we must reshape our data points to be vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Double checking to make sure the number of data points equals the number of labels
assert x_train.shape[0] == y_train.shape[0]
assert x_test.shape[0] == y_test.shape[0]

###########################
#Building TensorFlow Graph#
###########################

with tf.name_scope("Training_Dataset"):
    # Setting up placeholder objects to feed data to for the training dataset
    x_train_placeholder = tf.placeholder(x_train.dtype, x_train.shape)
    y_train_placeholder = tf.placeholder(y_train.dtype, y_train.shape)
    
    # Setting up Dataset objects to store data using unfed placeholder objects for the training dataset
    data_x_train = tf.data.Dataset.from_tensor_slices(x_train_placeholder)
    data_y_train = tf.data.Dataset.from_tensor_slices(y_train_placeholder).map(lambda el: tf.one_hot(el, 10))
    
    # Setting up training Dataset object from the previous Dataset objects
    training_dataset = tf.data.Dataset.zip((data_x_train, data_y_train)).shuffle(500).batch(training_batch_size)
    
    data_type = training_dataset.output_types
    data_shape = training_dataset.output_shapes
   
with tf.name_scope("Testing_Dataset"):
    # Setting up placeholder objects to feed data to for the testing dataset
    x_test_placeholder = tf.placeholder(x_test.dtype, x_test.shape)
    y_test_placeholder = tf.placeholder(y_test.dtype, y_test.shape)
    
    # Setting up Dataset objects to store data using unfed placeholder objects for the testing dataset
    data_x_test = tf.data.Dataset.from_tensor_slices(x_test_placeholder)
    data_y_test = tf.data.Dataset.from_tensor_slices(y_test_placeholder).map(lambda el: tf.one_hot(el,10))
    
    # Setting up testing Dataset object from the previous Dataset objects
    testing_dataset = tf.data.Dataset.zip((data_x_test, data_y_test)).shuffle(500).batch(32)

with tf.name_scope("Dataset_Iterators"):
    # Creating a general Iterator that can switch between consuming different datasets through reinitialization
    iterator = tf.data.Iterator.from_structure(data_type, data_shape)
    
    # Initializing the general Iterator on different datasets
    #  - When traing run the training_init_op command of the graph
    #  - When testing run the testing_init_op command of the graph
    training_init_op = iterator.make_initializer(training_dataset)
    testing_init_op = iterator.make_initializer(testing_dataset)
    
# Defining NN Model
def neural_network(input, name = "network"):
    with tf.name_scope(name):
        # Weights connecting layers 1 (input) --> 2
        Theta_1 = tf.Variable(tf.random_normal([784, 50], stddev = 0.05), name = "Theta_1")
        Bias_1 = tf.Variable(tf.random_normal([50], stddev = 0.5), name = "Bias_1")
        # Weights connecting layers 2 --> 3
        Theta_2 = tf.Variable(tf.random_normal([50, 25], stddev = 0.05), name = "Theta_2")
        Bias_2 = tf.Variable(tf.random_normal([25], stddev = 0.5), name = "Bias_2")
        # Weights connecting layers 3 --> 4 (output)
        Theta_3 = tf.Variable(tf.random_normal([25, 10], stddev = 0.05), name = "Theta_3")
        Bias_3 = tf.Variable(tf.random_normal([10]), name = "Bias_3")
        # Calculating the activation of the layers
        a2 = tf.add(tf.matmul(input, Theta_1), Bias_1)
        a2 = tf.nn.relu(a2)
        a3 = tf.add(tf.matmul(a2, Theta_2), Bias_2)
        a3 = tf.nn.relu(a3)
        output = tf.add(tf.matmul(a3, Theta_3), Bias_3)
        return output

images, labels = iterator.get_next()
images = tf.cast(images, tf.float32)

# Building NN Model
y = neural_network(images)

# Building cost function (cross entropy) and training operation
with tf.name_scope("Cost_Function"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = y))

with tf.name_scope("Train"):
    train = tf.train.AdamOptimizer().minimize(cost)

# Calculating predictions and accuracy numbers
with tf.name_scope("Accuracy"):
    prediction = tf.argmax(y, 1)
    equality = tf.equal(prediction, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

init_op = tf.global_variables_initializer()

########################################
#Executing The Default TensorFlow Graph#
########################################

with tf.Session() as sess:
    # Writing TensorFlow graph to memory
    writer = tf.summary.FileWriter("D:\\Programs\\testing\\tf_tutorials\\NN_tutorial_logs")
    writer.add_graph(sess.graph)
    
    # Running Initilization operation of TensorFlow graph
    sess.run(init_op)

    # Training Run
    for i in range(epochs):
        sess.run(training_init_op, feed_dict={x_train_placeholder: x_train, y_train_placeholder: y_train})
        avg_cost = 0
        avg_acc = 0
        batch = 1
        while True:
            try:
                C, _, acc = sess.run([cost, train, accuracy])
                avg_cost = avg_cost + C
                avg_acc = avg_acc + acc
                if batch % 10 == 0:
                    print("Epoch: {}, Batch: {}, Cost: {:.3f}, training accuracy: {:.2f}%".format(i, batch, C, acc * 100))
                batch = batch + 1
            except tf.errors.OutOfRangeError:
                break
        avg_cost = avg_cost / batch
        avg_acc = avg_acc / batch 
        print("Epoch: {}, Average Cost: {:.3f}, average training accuracy: {:.2f}%".format(i, avg_cost, avg_acc * 100))
        
    # Testing Run
    sess.run(testing_init_op, feed_dict={x_test_placeholder: x_test, y_test_placeholder: y_test})
    batch = 1
    avg_acc = 0
    while True:
        try:
            acc = sess.run([accuracy])
            avg_acc = avg_acc + acc[0]
            batch = batch + 1
        except tf.errors.OutOfRangeError:
            break
    avg_acc = avg_acc / batch
    print("Average testing set accuracy over {} batches is {:.2f}%".format(batch, avg_acc * 100))
    
    
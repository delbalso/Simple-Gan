
# coding: utf-8

# # Basic Generative Adversarial Network to generate MNIST images
# 
# This code creates two networks: a generator and a discriminator. The discriminator tries to determine if it's input is coming from real MNIST data or not and the generator tries to fool the discriminator. The generator (which is the network that is making the images seen below) never actually sees data from the training set, it just connects to the discriminator and has a loss function based on the discriminator's output.
# 

# In[ ]:

from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from six.moves import range
import pylab as pl
import time
from IPython import display

get_ipython().magic(u'matplotlib inline')


# In[ ]:

pl.rcParams["figure.figsize"] = 15,15

def plot_images(session,image_graph):
    images = session.run([image_graph])
    images = np.squeeze(images)
    edge_size = np.sqrt(images.shape[0])
    new_row = []
    for i in xrange(images.shape[0]):
        new_row.append(images[i])
        if (i + 1) % edge_size == 0:
            if i < edge_size: #full_image doesn't exist yet
                full_image = np.hstack(new_row)
            else:
                full_image = np.vstack([full_image,np.hstack(new_row)])
            new_row = []
    pl.imshow(full_image,cmap="gray")
    display.display(pl.gcf())
    time.sleep(1.0)


# In[ ]:

import shutil
shutil.rmtree('/tmp/train')


# In[ ]:

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)


# In[ ]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# In[ ]:

batch_size = 64
patch_size = 5
depth = 16
num_hidden = 64

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

random_vector_size = 64

graph = tf.Graph()

with graph.as_default():
    global_step = tf.Variable(0, trainable=False)
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels)) #images
    tf_train_random = tf.placeholder(tf.float32, shape=(batch_size, random_vector_size)) #numerals

    # Variables for generator
    hidden_layer1_size = 50
    with tf.name_scope("normal_generator"):
        glayer1_weights = tf.Variable(tf.truncated_normal([random_vector_size,hidden_layer1_size], stddev=0.1), name="w1")
        glayer2_weights = tf.Variable(tf.truncated_normal([hidden_layer1_size,image_size * image_size], stddev=0.1), name="w2")
        glayer1_biases = tf.Variable(tf.zeros([hidden_layer1_size]), name="b1")
        glayer2_biases = tf.Variable(tf.zeros([image_size * image_size]), name="b2")
    
    # Variables for deconv generator
    with tf.name_scope("generator"):
        deconv_filter_1 = tf.Variable(tf.truncated_normal([5,5,256,1], stddev=0.1), name="filters/1")
        deconv_filter_2 = tf.Variable(tf.truncated_normal([8,8,64,256], stddev=0.1), name="filters/2")
        deconv_filter_3 = tf.Variable(tf.truncated_normal([10,10,1,64], stddev=0.1), name="filters/3")
        deconv_bias_1 = tf.Variable(tf.zeros([256]), name="biases1")
        deconv_bias_2 = tf.Variable(tf.zeros([64]), name="biases2")
        deconv_bias_3 = tf.Variable(tf.zeros([1]), name="biases3")

    # Variables for classifier
    with tf.name_scope("discriminator"):
        layer1_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, num_channels, depth], stddev=0.1), name="w1")
        layer2_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1), name="w2")
        layer3_weights = tf.Variable(tf.truncated_normal(
          [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1), name="w3")
        layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, 1], stddev=0.1), name="w4")
        layer1_biases = tf.Variable(tf.zeros([depth]), name="b1")
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]), name="b2")
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name="b3")
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[1]), name="b4")

    # Model.
    def classifier_model(data):
        with graph.as_default():
            with tf.name_scope("discriminator"):
                conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME', name="layer1")
                hidden = tf.nn.relu(conv + layer1_biases)
                conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME', name="layer2")
                hidden = tf.nn.relu(conv + layer2_biases)
                shape = tf.shape(hidden)
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases, name="layer3")
                return (tf.matmul(hidden, layer4_weights) + layer4_biases)

    #def generator_model(labels):
    #    hidden1 = tf.nn.relu(tf.matmul(labels, glayer1_weights) + glayer1_biases)
    #    hidden2 = tf.sigmoid(tf.matmul(hidden1, glayer2_weights) + glayer2_biases)
    #    return tf.reshape(hidden2,[-1,image_size, image_size,1])
    
    def deconv_generator_model(noise):
        square_noise = tf.reshape(noise, [-1,8,8,1])
        num_examples = tf.shape(square_noise)[0]
        with tf.name_scope("generator"):
            fm1 = tf.nn.conv2d_transpose(value=square_noise, #[10,8,8,1]
                                         filter=deconv_filter_1, #[5,5,64,1]
                                         output_shape=[num_examples,12,12,256],
                                         strides=[1,1,1,1],
                                         padding='VALID', name="fm1")
            layer1 = tf.nn.relu(tf.nn.bias_add(fm1,deconv_bias_1), name="layer1")
            print("layer1 shape= {0}".format(layer1.get_shape()))
            fm2 = tf.nn.conv2d_transpose(value=layer1,
                                         filter=deconv_filter_2,
                                         output_shape=[num_examples,19,19,64],
                                         strides=[1,1,1,1],
                                         padding='VALID', name="fm2")
            print("deconv bias2 shape= {0}".format(deconv_bias_2.get_shape()))
            layer2 = tf.nn.relu(tf.nn.bias_add(fm2,deconv_bias_2), name="layer2")
            print("layer2 shape= {0}".format(layer2.get_shape()))
            fm3 = tf.nn.conv2d_transpose(value=layer2,
                                         filter=deconv_filter_3,
                                         output_shape=[num_examples,28,28,1],
                                         strides=[1,1,1,1],
                                         padding='VALID', name="fm3")
            layer3 = tf.nn.relu(tf.nn.bias_add(fm3,deconv_bias_3),name="layer3")
            print("layer3 shape= {0}".format(layer3.get_shape()))
            output = tf.reshape(layer3,[-1,image_size, image_size,1])
            print("output shape= {0}".format(output.get_shape()))
        return output

    #Generator Loss
    sample_points = tf.constant(np.random.uniform(0,1,(batch_size,random_vector_size)).astype(np.float32))
    test_image = deconv_generator_model(sample_points)
    generated_image = deconv_generator_model(tf_train_random)
    
    generator_logits = classifier_model(generated_image)
    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(generator_logits, tf.ones([batch_size,1])))

    # Generator Optimizer
    generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
    generator_learnrate = tf.train.exponential_decay(0.05, global_step,
                                           1000, 0.96, staircase=True)
    generator_optimizer = tf.train.AdagradOptimizer(generator_learnrate).minimize(generator_loss, var_list=generator_variables, global_step=global_step)  

    # Discriminator Loss
    classifier_real_logits = classifier_model(tf_train_dataset)
    classifier_real_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(classifier_real_logits, tf.ones([batch_size,1])))

    classifier_fake_logits = classifier_model(generated_image)
    classifier_fake_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(classifier_fake_logits, tf.zeros([batch_size,1])))

    classifier_loss = classifier_real_loss + classifier_fake_loss

    # Discriminator Optimizer.
    classifier_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
    classifier_learnrate = tf.train.exponential_decay(0.05, global_step,
                                           1000, 0.96, staircase=True)
    classifier_optimizer = tf.train.AdagradOptimizer(classifier_learnrate).minimize(classifier_loss, var_list=classifier_variables)
    
    # Log variables
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"):
        variable_summaries(var, var.name)
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator"):
        variable_summaries(var, var.name)

    tf.image_summary(deconv_filter_1.name,deconv_filter_1)
    merged = tf.merge_all_summaries()



# In[ ]:

num_steps = 70000
training_thresh = 0.4
updating = 'discriminator'
step = 0
updated_generator=True
l1,l2,l3 = .5, .5, .5

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    train_writer = tf.train.SummaryWriter('/tmp/train', session.graph)
    print('Initialized')

    for step in range(num_steps):
        # prepare batch of training data
        batch_data, batch_labels = mnist.train.next_batch(batch_size)
        batch_data = batch_data.reshape([-1,28,28,1])
        feed_dict = {tf_train_dataset : batch_data, tf_train_random: np.random.normal(0,1.0/np.sqrt(64),(batch_size,random_vector_size)).astype(np.float32)}

        # decide if we should change which model to update
        if float(l3)<training_thresh and updating=='generator':
            print("Updating discriminator...")
            updating='discriminator'
        elif float(l2+l1)<training_thresh and updating=='discriminator':
            print("Updating generator...")
            updating='generator'
        
        if updating == 'generator':
            updated_generator=True

        # update model
        #if updating=='discriminator':
        #    _ = session.run([classifier_optimizer], feed_dict=feed_dict)
        #elif(updating=='generator'):
        #    _ = session.run([generator_optimizer], feed_dict=feed_dict)
        #if l3<l1+l2:
        _ = session.run([classifier_optimizer], feed_dict=feed_dict)
        #else:
        summary, _, gs = session.run([merged, generator_optimizer, global_step], feed_dict=feed_dict)
        updated_generator=True
        train_writer.add_summary(summary, gs)
        #    _ = session.run([generator_optimizer], feed_dict=feed_dict)
        #    updated_generator=True

        if (step % 100 == 0):
            # log/debug    
            images, l3,l1, l2 = session.run([generated_image, generator_loss, classifier_real_loss, classifier_fake_loss], feed_dict=feed_dict)
            print("Step {0}".format(step))
            print("Real and Fake loss: {0}".format([l1,l2]))
            print("Generator Loss {0}".format(l3))
            clr, glr = session.run([classifier_learnrate,generator_learnrate])
            print ("Classifier learn rate: {0}, Generator learn rate: {1}".format(clr,glr))
            
            if updated_generator:
                plot_images(session,test_image)
                updated_generator=False


# In[ ]:




# In[ ]:




# In[ ]:




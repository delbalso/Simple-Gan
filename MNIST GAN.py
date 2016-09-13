
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

import os
import sys

STDOUT = 1
STDERR = 2

class FDRedirector(object):
    """ Class to redirect output (stdout or stderr) at the OS level using
        file descriptors.
    """ 

    def __init__(self, fd=STDOUT):
        """ fd is the file descriptor of the outpout you want to capture.
            It can be STDOUT or STERR.
        """
        self.fd = fd
        self.started = False
        self.piper = None
        self.pipew = None

    def start(self):
        """ Setup the redirection.
        """
        if not self.started:
            self.oldhandle = os.dup(self.fd)
            self.piper, self.pipew = os.pipe()
            os.dup2(self.pipew, self.fd)
            os.close(self.pipew)

            self.started = True

    def flush(self):
        """ Flush the captured output, similar to the flush method of any
        stream.
        """
        if self.fd == STDOUT:
            sys.stdout.flush()
        elif self.fd == STDERR:
            sys.stderr.flush()

    def stop(self):
        """ Unset the redirection and return the captured output. 
        """
        if self.started:
            self.flush()
            os.dup2(self.oldhandle, self.fd)
            os.close(self.oldhandle)
            f = os.fdopen(self.piper, 'r')
            output = f.read()
            f.close()

            self.started = False
            return output
        else:
            return ''

    def getvalue(self):
        """ Return the output captured since the last getvalue, or the
        start of the redirection.
        """
        output = self.stop()
        self.start()
        return output


# In[ ]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# In[ ]:

def maxpool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv (input_data, shape, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", [shape[3]], initializer=tf.truncated_normal_initializer(stddev=0.1))
        return tf.nn.conv2d(input_data, w, [1, 2, 2, 1], padding='SAME')+b


def fully_connected(input_data, shape, name="fully_connected"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", [shape[1]], initializer=tf.truncated_normal_initializer(stddev=0.1))
    return tf.matmul(input_data, w) + b

def discriminator(data, reuse=True):
    with tf.variable_scope("discriminator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        channels = 32
        d_l1 = maxpool(conv(data, shape=[3, 3, num_channels, channels], name="layer1"))
        d_l2 = conv(tf.nn.relu(d_l1), shape=[3, 3, channels, channels/2], name="layer2")
        d_l3 = conv(tf.nn.relu(d_l2), shape=[3, 3, channels/2, channels/4], name="layer3")
        d_l3_reshaped = tf.reshape(d_l3, [tf.shape(d_l3)[0],2*2*8])
        fm3_shape = 2*2*channels/4
        d_l4 = fully_connected(tf.nn.relu(d_l3_reshaped), shape=[fm3_shape,fm3_shape/2], name="layer4")
        d_l5 = fully_connected(d_l4, shape=[fm3_shape/2,1], name="layer5")
        print ("d_l5 {0}".format(d_l5.get_shape()))
        #output = tf.matmul(reshape,w) + b
        return tf.Print(tf.nn.sigmoid(d_l5),[tf.shape(d_l5)], first_n=1)


# In[ ]:

batch_size = 64
num_hidden = 64

image_size = 28
num_channels = 1 # grayscale

random_vector_size = 64

graph = tf.Graph()

with graph.as_default():
    global_step = tf.Variable(0, trainable=False)
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels)) #images
    tf_train_random = tf.placeholder(tf.float32, shape=(batch_size, random_vector_size))

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
        deconv_offset_1 = tf.Variable(tf.truncated_normal([256], stddev=0.1), name="offset1")
        deconv_offset_2 = tf.Variable(tf.truncated_normal([64], stddev=0.1), name="offset2")
        deconv_offset_3 = tf.Variable(tf.truncated_normal([1], stddev=0.1), name="offset3")
        deconv_scale_1 = tf.Variable(tf.truncated_normal([256], stddev=0.1), name="scale1")
        deconv_scale_2 = tf.Variable(tf.truncated_normal([64], stddev=0.1), name="scale2")
        deconv_scale_3 = tf.Variable(tf.truncated_normal([1], stddev=0.1), name="scale3")

  
    def batch_norm(inputs, scale, offset):
        mean, variance = tf.nn.moments(inputs,axes=[0,1,2])
        return tf.nn.batch_normalization(inputs, mean, variance, offset, scale, variance_epsilon=1e-5)
    
    def deconv_generator_model(noise):
        square_noise = tf.reshape(noise, [-1,8,8,1])
        num_examples = tf.shape(square_noise)[0]
        with tf.name_scope("generator"):
            fm1 = tf.nn.conv2d_transpose(value=square_noise, #[10,8,8,1]
                                         filter=deconv_filter_1, #[5,5,64,1]
                                         output_shape=[num_examples,12,12,256],
                                         strides=[1,1,1,1],
                                         padding='VALID', name="fm1")
            layer1 = tf.nn.relu(batch_norm(fm1,deconv_scale_1, deconv_offset_1))
            print("layer1 shape= {0}".format(layer1.get_shape()))
            fm2 = tf.nn.conv2d_transpose(value=layer1,
                                         filter=deconv_filter_2,
                                         output_shape=[num_examples,19,19,64],
                                         strides=[1,1,1,1],
                                         padding='VALID', name="fm2")
            #print("deconv bias2 shape= {0}".format(deconv_bias_2.get_shape()))
            layer2 = tf.nn.relu(batch_norm(fm2,deconv_scale_2, deconv_offset_2))
            print("layer2 shape= {0}".format(layer2.get_shape()))
            fm3 = tf.nn.conv2d_transpose(value=layer2,
                                         filter=deconv_filter_3,
                                         output_shape=[num_examples,28,28,1],
                                         strides=[1,1,1,1],
                                         padding='VALID', name="fm3")
            layer3 = tf.nn.relu(batch_norm(fm3,deconv_scale_3, deconv_offset_3))
            print("layer3 shape= {0}".format(layer3.get_shape()))
            output = tf.reshape(layer3,[-1,image_size, image_size,1])
            print("output shape= {0}".format(output.get_shape()))
        return output

    #Generator Loss
    sample_points = tf.constant(np.random.uniform(0,1,(batch_size,random_vector_size)).astype(np.float32))
    debug_image = deconv_generator_model(sample_points)
    generated_image = deconv_generator_model(tf_train_random)
    
    generator_logits = discriminator(generated_image, reuse=False)
    generator_loss = tf.nn.l2_loss(generator_logits- tf.ones([batch_size,1]))

    # Generator Optimizer
    generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
    generator_learnrate = tf.train.exponential_decay(0.05, global_step, 1000, 0.96, staircase=True)
    generator_optimizer = tf.train.AdagradOptimizer(generator_learnrate).minimize(generator_loss, var_list=generator_variables, global_step=global_step)  

    # Discriminator Loss
    classifier_real_logits = discriminator(tf_train_dataset)
    print(tf_train_dataset.get_shape())
    print(classifier_real_logits.get_shape())
    classifier_real_loss = tf.nn.l2_loss(classifier_real_logits - tf.ones([batch_size,1]))

    classifier_fake_logits = generator_logits
    classifier_fake_loss = tf.nn.l2_loss(classifier_fake_logits - tf.zeros([batch_size,1]))

    classifier_loss = classifier_real_loss + classifier_fake_loss

    # Discriminator Optimizer.
    classifier_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
    classifier_learnrate = tf.train.exponential_decay(0.05, global_step, 1000, 0.96, staircase=True)
    classifier_optimizer = tf.train.AdagradOptimizer(classifier_learnrate).minimize(classifier_loss, var_list=classifier_variables, global_step=global_step)
    d_opt = tf.train.AdagradOptimizer(classifier_learnrate)
    d_gradients = d_opt.compute_gradients(classifier_loss, classifier_variables)
    d_apply = d_opt.apply_gradients(d_gradients, global_step=global_step)
    
    # Log variables
    for var in classifier_variables:
        print(var.name)
        variable_summaries(var, var.name)
    for var in generator_variables:
        variable_summaries(var, var.name)
    variable_summaries(classifier_real_logits, "discriminator/real_logits")
    variable_summaries(classifier_fake_logits, "discriminator/fake_logits")
    #variable_summaries(tf.nn.sigmoid_cross_entropy_with_logits(classifier_real_logits, tf.ones([batch_size,1])), "discriminator/cross_entropy_real_logits")
    #variable_summaries(tf.nn.sigmoid_cross_entropy_with_logits(classifier_fake_logits, tf.zeros([batch_size,1])), "discriminator/cross_entropy_fake_logits")
    variable_summaries(generator_logits, "generator/logits")
        
        
    check = tf.add_check_numerics_ops()

    tf.image_summary(deconv_filter_1.name,deconv_filter_1)
    merged = tf.merge_all_summaries()



# In[ ]:

num_steps = 70000
training_thresh = 0.4
updating = 'discriminator'
step = 0
updated_generator=True
l1,l2,l3 = .5, .5, .5
redirect=FDRedirector(STDERR)

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

        # update model
        #if updating=='discriminator':
        #    _ = session.run([classifier_optimizer], feed_dict=feed_dict)
        #elif(updating=='generator'):
        #    _ = session.run([generator_optimizer], feed_dict=feed_dict)
        
        
        #if l3<l1+l2:
        #redirect.start()
        _ = session.run([check], feed_dict=feed_dict)
        summary, grad, _,gs = session.run([merged, d_gradients, d_apply, global_step], feed_dict=feed_dict)
        #classifier_real_logits.eval(feed_dict=feed_dict)
        #summary, _ = session.run([merged, generator_optimizer], feed_dict=feed_dict)
        _ = session.run([generator_optimizer], feed_dict=feed_dict)
        _ = session.run([generator_optimizer], feed_dict=feed_dict)
        _ = session.run([generator_optimizer], feed_dict=feed_dict)
        #print (redirect.stop())
        #else:
        #    _ = session.run([generator_optimizer], feed_dict=feed_dict)
        #    _ = session.run([generator_optimizer], feed_dict=feed_dict)
        #summary, _, gs = session.run([merged, generator_optimizer, global_step], feed_dict=feed_dict)

        #raise
        updated_generator=True
        train_writer.add_summary(summary, gs)
        #    _ = session.run([generator_optimizer], feed_dict=feed_dict)
        #    updated_generator=True

        if (step % 100 == 0):
            #if step >1000:
            #    raise
            #for var in classifier_variables:
            #    print("Name: {0}, Shape: {1}".format(var.name,tf.shape(var).eval()))
            assert (len(grad)==len(classifier_variables))
            for i in xrange(len(classifier_variables)):
                g = grad[i][0]
                print ("{0}'s gradients have {1} ({2}%) zeros and mean {3}".format(classifier_variables[i].name,g.size-np.count_nonzero(g), (g.size-np.count_nonzero(g)+0.0)/g.size*100, np.mean(g)))
            #for gradient, var in grad:
            #    print("Variable: {0}\n------gradient: {1}".format(var, gradient))
            
            
            # log/debug    
            images, l3,l1, l2 = session.run([generated_image, generator_loss, classifier_real_loss, classifier_fake_loss], feed_dict=feed_dict)
            print("Step {0}".format(step))
            print("Classifier loss: {0}, Real: {1}, Fake: {2}".format(l1+l2, l1,l2))
            print("Generator Loss: {0}".format(l3))
            clr, glr = session.run([classifier_learnrate,generator_learnrate])
            print ("Classifier learn rate: {0}, Generator learn rate: {1}".format(clr,glr))
            
            if updated_generator:
                plot_images(session,debug_image)
                updated_generator=False


# In[ ]:




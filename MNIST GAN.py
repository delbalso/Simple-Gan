
# coding: utf-8

# # Basic Generative Adversarial Network to generate MNIST images
# 
# This code creates two networks: a generator and a discriminator. The discriminator tries to determine if it's input is coming from real MNIST data or not and the generator tries to fool the discriminator. The generator (which is the network that is making the images seen below) never actually sees data from the training set, it just connects to the discriminator and has a loss function based on the discriminator's output.
# 

# In[ ]:

#from __future__ import print_function
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#from six.moves import range
import pylab as pl
import time
from IPython import display
#import tdb
#from tdb.examples import viz

get_ipython().magic(u'matplotlib inline')


# In[ ]:

import PIL.Image as im

def save_image(image):
    im.fromarray(np.uint8(pl.cm.Greys(image)*255)).convert('RGB').save("result.jpeg")
    

pl.rcParams["figure.figsize"] = 15,15

def plot_images(image_ref, session=None, tensor=True):
    if tensor:
        images = session.run([image_ref])
    else:
        images = image_ref
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
    save_image(full_image)
    pl.imshow(full_image,cmap="gray_r")
    
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
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv(input_data, shape, k=2, name="conv2d"):
    with tf.variable_scope(name):
        filter = tf.get_variable("filter", shape, initializer=tf.truncated_normal_initializer(stddev=1))
        b = tf.get_variable("b", [shape[3]], initializer=tf.truncated_normal_initializer(stddev=0.01))
        return tf.nn.conv2d(input_data, filter, [1, k, k, 1], padding='SAME')+b


def fully_connected(input_data, shape, use_bias = True, stddev = 0.01, name="fully_connected"):
    assert input_data.get_shape()[1] == shape[0], "input shape = {0}, w shape = {1}".format(input_data.get_shape(),shape)
    with tf.variable_scope(name):
        w = tf.get_variable("w", shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        if use_bias:
            b = tf.get_variable("b", [shape[1]], initializer=tf.truncated_normal_initializer(stddev=stddev))
            return tf.matmul(input_data, w) + b
        else:
            return tf.matmul(input_data, w)

def minibatch(input, num_kernels=5, kernel_dim=3, name="minibatch"):
    x = fully_connected(input, [input.get_shape()[1], num_kernels * kernel_dim], use_bias=False, stddev = 100, name=name)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) -         tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat(1, [minibatch_features, input])

def discriminator(data, reuse=True):
    with tf.variable_scope("discriminator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        channels = 4
        d_l1 = batch_norm(conv(data, shape=[4, 4, num_channels, channels], k=2, name="layer1"), channels/1, name = "layer1")
        d_l2 = batch_norm(conv(tf.nn.relu6(d_l1), shape=[4, 4, channels, channels/2], k=2,name="layer2"), channels/2, name = "layer2")
        d_l3 = batch_norm(conv(tf.nn.relu6(d_l2), shape=[4, 4, channels/2, channels/4], k=2, name="layer3"), channels/4, name = "layer3")
        fm3_shape = 4*4*channels/4
        d_l3_reshaped = tf.reshape(d_l3, [tf.shape(d_l3)[0],fm3_shape])
        #data_reshaped = tf.reshape(data, [64,28*28])
        
        mb = minibatch(d_l3_reshaped, name="layer3_minibatch")
        print("mb shape {0}".format(mb.get_shape()))
        d_l4 = fully_connected(tf.nn.relu6(mb), shape=[mb.get_shape()[1],2], name="layer4")
        #return tf.Print(tf.nn.sigmoid(d_l4),[mb[:,0]], message = "MB's Value: ")
        return d_l4#tf.nn.sigmoid(d_l4)
    
def deconv2d(input_data, filter_shape, output_shape, padding = 'SAME', k = 2, name="deconv2d"):
    with tf.variable_scope(name):
        filter = tf.get_variable("filter", filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
        return tf.nn.conv2d_transpose(value=input_data,
                                     filter=filter, 
                                     output_shape=output_shape,
                                     strides=[1,k,k,1],
                                     padding=padding, name="feature_maps")
    
def batch_norm(input_data, size,name="batch_norm"):
    with tf.variable_scope(name):
        offset = tf.get_variable("offset", [size], initializer=tf.truncated_normal_initializer(stddev=0.01))
        scale = tf.get_variable("scale", [size], initializer=tf.truncated_normal_initializer(stddev=0.01))
        mean, variance = tf.nn.moments(input_data,axes=[0,1,2])
        return tf.nn.batch_normalization(input_data, mean, variance, offset, scale, variance_epsilon=1e-5)
    
def generator(noise, reuse=True):
    num_examples = tf.shape(noise)[0]
    with tf.variable_scope("generator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        l0 = fully_connected(noise, shape=[64,4*4*512], name="layer0")
        l0_sq = tf.reshape(l0, [num_examples,4,4,512])
        layer0 =  tf.nn.relu(batch_norm(l0_sq,512, name = "layer0"))
        
        
        fm1 = deconv2d(layer0, # 64,8,8,1
                      filter_shape= [4,4,256,512], 
                      output_shape=[num_examples,7,7,256],
                      padding = "VALID",
                      k=1,
                      name = "layer1")
        layer1 = tf.nn.relu(batch_norm(fm1,256, name = "layer1"))
        
        fm2 = deconv2d(layer1,
                      filter_shape= [4,4,128,256],
                      output_shape=[num_examples,14,14,128],
                      name = "layer2")
        layer2 = tf.nn.relu(batch_norm(fm2,128, name = "layer2"))

        fm3 = deconv2d(layer2,
                      filter_shape=[4,4,1,128],
                      output_shape=[num_examples,28,28,1],
                      name = "layer3")

        layer3 = tf.nn.sigmoid(fm3)
        output = layer3#tf.reshape(layer4,[-1,image_size, image_size,1])
        print("layer1 shape= {0}".format(layer1.get_shape()))
        print("layer2 shape= {0}".format(layer2.get_shape()))
        print("layer3 shape= {0}".format(layer3.get_shape()))
#        print("layer4 shape= {0}".format(layer4.get_shape()))
        print("output shape= {0}".format(output.get_shape()))
    return output


# In[ ]:

batch_size = 64
image_size = 28
num_channels = 1 # grayscale

random_vector_size = 64

tf.reset_default_graph()
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

    #Generator Loss
    sample_points = tf.constant(np.random.uniform(0,1,(batch_size,random_vector_size)).astype(np.float32))
    debug_image = generator(sample_points, reuse=False)
    generated_image = generator(tf_train_random)
    
    generator_logits = discriminator(generated_image, reuse=False)
    generator_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(generator_logits,tf.tile(tf.constant([[0,1]],dtype=tf.float32),[batch_size,1])))
    
    generator_loss2 = tf.nn.l2_loss(generated_image-tf_train_dataset)
    

    # Generator Optimizer
    generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
    generator_learnrate = tf.train.exponential_decay(0.003, global_step, 500, 0.96, staircase=True)
    generator_optimizer = tf.train.AdamOptimizer(generator_learnrate).minimize(generator_loss, var_list=generator_variables, global_step=global_step)  

    generator_optimizer2 = tf.train.AdamOptimizer(.003).minimize(generator_loss2, var_list=generator_variables, global_step=global_step)  
    
    # Discriminator Loss
    classifier_real_logits = discriminator(tf_train_dataset)
    print(tf_train_dataset.get_shape())
    print(classifier_real_logits.get_shape())
    print(generator_logits.get_shape())
    classifier_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(classifier_real_logits,tf.tile(tf.constant([[0,1]],dtype=tf.float32),[batch_size,1])))

    classifier_fake_logits = generator_logits
    classifier_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(classifier_fake_logits,tf.tile(tf.constant([[1,0]],dtype=tf.float32),[batch_size,1])))

    classifier_loss = classifier_real_loss + classifier_fake_loss

    # Discriminator Optimizer.
    classifier_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
    classifier_learnrate = tf.train.exponential_decay(0.003, global_step, 500, 0.96, staircase=True)
    classifier_optimizer = tf.train.AdamOptimizer(classifier_learnrate).minimize(classifier_loss, var_list=classifier_variables, global_step=global_step)
    d_opt = tf.train.AdamOptimizer(classifier_learnrate)
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
    variable_summaries(generator_logits, "generator/logits")
        
        
    check = tf.add_check_numerics_ops()
    #for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*filter"):
        #print ("Image variable: {0}".format(var.name))
        #tf.summary.tensor_summary(var.name + '_tensor',var)
    merged = tf.merge_all_summaries()
    #p0 = tdb.plot_op(viz.viz_conv_weights,inputs=[tf.get_default_graph().get_tensor_by_name("generator/layer1/filter:0")])



# In[ ]:

num_steps = 70000
step = 0
updated_generator=True
cr_loss, cf_loss, g_loss = .5, .5, .5
redirect=FDRedirector(STDERR)
#batch_data, batch_labels = mnist.train.next_batch(batch_size)
#batch_data = np.tile(batch_data[1:3], (batch_size/2,1))
#batch_data = batch_data.reshape([batch_size,28,28,1])
#plot_images(batch_data, tensor=False)
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    train_writer = tf.train.SummaryWriter('/tmp/train', session.graph)
    print('Initialized')
    for step in xrange(num_steps):
        # prepare batch of training data
        batch_data, batch_labels = mnist.train.next_batch(batch_size)
        batch_data = batch_data.reshape([-1,28,28,1])
        feed_dict = {tf_train_dataset : batch_data, 
                     tf_train_random: np.random.random((batch_size,random_vector_size)).astype(np.float32)}

#        _ = tdb.debug([p0], feed_dict=feed_dict, session =tf.get_default_session())
        _ = session.run([check], feed_dict=feed_dict)
        #_ = session.run([generator_optimizer2], feed_dict=feed_dict)


        #if g_loss < cr_loss + cf_loss or step<300:
        """grad, = session.run([d_gradients], feed_dict=feed_dict)
        assert (len(grad)==len(classifier_variables))
        for i in xrange(len(classifier_variables)):
            g = grad[i][0]
            print (g)
            print ("{0}'s gradients have {1} ({2}%) zeros and mean {3}".format(classifier_variables[i].name,g.size-np.count_nonzero(g), (g.size-np.count_nonzero(g)+0.0)/g.size*100, np.mean(g)))
        
        
        raise"""
        grad, _ = session.run([d_gradients, d_apply], feed_dict=feed_dict)

        
        #else:
        if step>=100:
            _ = session.run([generator_optimizer], feed_dict=feed_dict)
            _ = session.run([generator_optimizer], feed_dict=feed_dict)
            updated_generator=True

        summary, gs = session.run([merged, global_step], feed_dict=feed_dict)

        
        train_writer.add_summary(summary, gs)
        #redirect.start()
        g_loss, cr_loss, cf_loss = session.run([generator_loss,
                                                classifier_real_loss,
                                                classifier_fake_loss], feed_dict=feed_dict)
        #print (redirect.stop())
        
        if (step % 10 == 0 or step <100):
            print("Generator Loss: {0}, Classifier loss: {1}, Real: {2}, Fake: {3}".format(g_loss, cr_loss + cf_loss, cr_loss, cf_loss))
        if (step % 100 == 0):
            
            images = session.run([generated_image], feed_dict=feed_dict)
            #if step >1000:
            #    raise
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator/layer3_minibatch"):
                print("Name: {0}, Shape: {1}".format(var.name,tf.shape(var).eval()))
            assert (len(grad)==len(classifier_variables))
            for i in xrange(len(classifier_variables)):
                g = grad[i][0]
                print ("{0}'s gradients have {1} ({2}%) zeros and mean {3}".format(classifier_variables[i].name,g.size-np.count_nonzero(g), (g.size-np.count_nonzero(g)+0.0)/g.size*100, np.mean(g)))
            #print(tf.get_default_graph().get_tensor_by_name("discriminator/layer3_minibatch/w:0").eval()[0])
            
            
            print("Step {0}".format(step))
            #print("Classifier loss: {0}, Real: {1}, Fake: {2}".format(cr_loss + cf_loss, cr_loss, cf_loss))
            #print("Generator Loss: {0}".format(g_loss))
            clr, glr = session.run([classifier_learnrate,generator_learnrate])
            print ("Classifier learn rate: {0}, Generator learn rate: {1}".format(clr,glr))
            
            if updated_generator:
                plot_images(debug_image, session=session)
                updated_generator=False


# In[ ]:




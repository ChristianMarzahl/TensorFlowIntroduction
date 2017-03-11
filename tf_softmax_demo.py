# Solution is available in the other "solution.py" tab
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    return np.divide(np.exp(x),np.sum(np.exp(x),axis=0))

logits = np.array([
    [1, 2, 3, 6],
    [2, 4, 5, 6],
    [3, 8, 7, 6]])
logits = np.multiply(logits,10)
print(softmax(logits))

logits = np.divide(logits,100)
print(softmax(logits))

import tensorflow as tf

logit_data = [2.0, 1.0, 0.1]
logits = tf.placeholder(tf.float32)
    
# TODO: Calculate the softmax of the logits
softmax = tf.nn.softmax(logits) 
    
with tf.Session() as sess:
    # TODO: Feed in the logit data
    output = sess.run(softmax, feed_dict={logits: logit_data})
    print (output)
# Solution is available in the other "solution.py" tab
import tensorflow as tf

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
z = tf.subtract(tf.div(x,y),tf.constant(1))

with tf.Session() as sess:
    output = sess.run(z, feed_dict={x: 10, y: 2})
    print (output)


x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x,y),tf.cast(tf.constant(1), tf.float64))

with tf.Session() as sess:
    output = sess.run(z)
    print(output)
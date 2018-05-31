import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import numpy as np


def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data(path="C:/Users/476458/.keras/datasets/mnist.npz")

    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test

def onehotrepresentation( dataset ) :
    onehotdataset = np.zeros((dataset.shape[0], 10))
    onehotdataset[np.arange(dataset.shape[0]), dataset] = 1
    return onehotdataset

def main():

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(True)
    print(X_train.shape, y_train.shape)

    y_train_onehot = onehotrepresentation(y_train)
    y_test_onehot = onehotrepresentation(y_test)

    X = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    initializer = tf.contrib.layers.xavier_initializer()
    W1 = tf.Variable(initializer(shape=[784, 140]),tf.float32,name="W1")
    b1 = np.zeros((1, 140))
    y1 = tf.add(tf.matmul(X,W1 ), b1)

    y1 = tf.layers.dropout(y1, 0.5)
    y1 = tf.nn.relu(y1)

    W2 = tf.Variable(initializer(shape=[140, 10]),tf.float32,name="W2")
    b2 = tf.Variable(tf.random_normal([10]))
    y2 = tf.add(tf.matmul(y1,W2 ), b2)


    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y2, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(
            init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        train_writer = tf.summary.FileWriter('D:/Development_Avecto/TensorFlow/logs/1/train', sess.graph)

        num = 0
        trainlosses =[]
        testlosses =[]
        for it in range(990):
            num = num + 50
            minibatch(X_train, y_train, num)
            x_batch, y_batch =  minibatch(X_train, y_train_onehot,num)

            _, loss_train = sess.run([optimizer,cross_entropy], feed_dict={ X: x_batch,y: y_batch})
            _, loss_test = sess.run([optimizer,cross_entropy], feed_dict={X: X_test, y: y_test_onehot})
            print ("Iterator " + str(it))
            trainlosses.append(loss_train)
            testlosses.append(loss_test)


        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y2, 1), tf.argmax(y, 1)), "float"))
        accuracy_value = sess.run(accuracy, feed_dict={X: X_test, y: onehotrepresentation(y_test)})
        print("Test Accuracy is " + str(accuracy_value))

        accuracy_value = sess.run(accuracy, feed_dict={X: X_val, y: onehotrepresentation(y_val)})
        print("Validation Accuracy is " + str(accuracy_value))

        plt.subplot(1, 2, 1)
        plt.plot(trainlosses)
        plt.title('Train Loss')
        plt.subplot(1, 2, 2)
        plt.plot(testlosses)
        plt.title('Test Loss')
        plt.show()
        coord.request_stop()
        coord.join(threads)


def minibatch( trainx, trainy, batch ) :
    x,y = (trainx[batch:batch + 50],trainy[batch:batch + 50])
    return (x,y)

if __name__ == "__main__":
    main()

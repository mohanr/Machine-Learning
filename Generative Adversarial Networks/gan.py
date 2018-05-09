import tensorflow as tf
import numpy as np
from PIL import Image
import os
import ntpath


# The part that discriminates
X = tf.placeholder(tf.float32, shape=[None, 299, 299, 1], name='X')


# The part that generates
Z = tf.placeholder(dtype=tf.float32,
                              shape=(None, 100),
                              name='Z')


def generator(z,reuse=False):
    with tf.variable_scope('generator',reuse=reuse):
        linear = tf.layers.dense(z, 1024 * 8 * 8)
        conv = tf.reshape(linear, (-1, 64, 64, 32))
        out = tf.layers.conv2d_transpose(conv, 64,kernel_size=4,strides=2, padding='SAME')
        out = tf.nn.relu(out)
        out = tf.layers.conv2d_transpose(out, 1,kernel_size=4,strides=2, padding='SAME')
        print( out.get_shape())
        out = tf.nn.relu(out)
        tf.nn.tanh(out)
        return out


def discriminator(x,reuse=False):
    with tf.variable_scope('discriminator',reuse=reuse):
        out = tf.layers.conv2d(x, filters=32, kernel_size=[5, 5], padding='SAME')
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, pool_size=[2, 2],padding='SAME', strides=2)
        out = tf.layers.conv2d(out, filters=64, kernel_size=[5, 5], padding='SAME')
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, pool_size=[2, 2],padding='SAME', strides=2)
        out = tf.layers.dense(out, units=256, activation=tf.nn.relu)
        out = tf.layers.dense(out, units=1, activation=tf.nn.sigmoid)
        return out


GeneratedImage = generator(Z)

DxL = discriminator(X)
DgL = discriminator(GeneratedImage, reuse=True)

D_Disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DxL, labels = tf.ones_like(DxL)))
D_Disc_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DgL, labels = tf.ones_like(DgL)))
D_MainLoss = D_Disc_loss + D_Disc_loss1

G_Generate_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DgL, labels = tf.ones_like(DgL)))

D_loss = tf.summary.scalar("Discriminator Loss", D_MainLoss)
G_loss = tf.summary.scalar("Generator Loss", G_Generate_loss)

variables = tf.trainable_variables()
dvariables = [var for var in variables if var.name.startswith("discriminator")]
gvariables = [var for var in variables if var.name.startswith("generator")]

D_optimizer = tf.train.AdamOptimizer().minimize(D_Disc_loss, var_list=dvariables)
G_optimizer = tf.train.AdamOptimizer().minimize(G_Generate_loss, var_list=gvariables)



def preprocess():
    filenames = tf.train.string_input_producer(
        tf.train.match_filenames_once("D:/Development_Avecto/TensorFlow/train/train/*.png"))

    reader = tf.WholeFileReader()
    key, value = reader.read(filenames)
    document = tf.image.decode_png(value, channels=3)
    document = resize(document)
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(50):
            file, document_tensor = sess.run([key, document])
            head,tail = ntpath.split(file)
            print(head)
            print(head.decode("utf-8") + "\\" + tail.decode("utf-8").split(".")[0] + "-resized.png")
            document_tensor = Image.fromarray(document_tensor, "RGB")
            document_tensor.save(os.path.join(head.decode("utf-8"), "\\" + tail.decode("utf-8").split(".")[0] + "-resized.png"))

        coord.request_stop()
        coord.join(threads)
        sess.close()

def resize(image):
    resized_image = tf.cast(tf.image.resize_images(image, [299, 299]), tf.uint8)
    return resized_image


def samplefromuniformdistribution(m, n):
    return np.random.uniform(-1., 1., size=(m, n))


def printtest():
    x = tf.constant([100.0])
    x = tf.Print(x,[x],message="Test")
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    b = tf.add(x, x)
    with tf.Session() as sess:
        sess.run(init)
        c = sess.run(b)
        print(c)
        sess.close()

def train():
    filenames = tf.train.string_input_producer(
        tf.train.match_filenames_once("D:/Development_Avecto/TensorFlow/resizedimages/*.png"))
    reader = tf.WholeFileReader()
    _, input = reader.read(filenames)
    #input = tf.Print(input,[input,tf.shape(input),"Input shape"])
    input = tf.image.decode_png(input, channels=1)
    input.set_shape([299, 299, 1])

    batch = tf.train.batch([input],
                           batch_size=20)


    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        train_writer = tf.summary.FileWriter('D:/Development_Avecto/TensorFlow/logs/1/train', sess.graph)
        tf.summary.image("Image", GeneratedImage)
        merge = tf.summary.merge_all()

        for it in range(500):
            _, X_batch =  sess.run([input,batch])
            summary,_ = sess.run([merge,D_optimizer], feed_dict={Z : samplefromuniformdistribution(20,100), X: X_batch})
            summary,_ = sess.run([merge,G_optimizer],feed_dict={ Z : samplefromuniformdistribution(20,100), X: X_batch})

            train_writer.add_summary(summary, it)
            train_writer.flush()

        train_writer.close()
        coord.request_stop()
        coord.join(threads)

def main():
    # os.chdir('D:/Development_Avecto/TensorFlow/train/train')
    #preprocess()
    #printtest()
    train()


if __name__ == "__main__":
    main()

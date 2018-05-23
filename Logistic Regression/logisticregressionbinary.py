import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing


data = pd.read_csv('D:/Development_Avector/PycharmProjects/TensorFlow/anes_dataset.csv')
training_features = ['TVnews', 'PID', 'age', 'educ', 'income']
target = 'vote'

print (data.columns)

X = data.loc[:, ['TVnews', 'PID', 'age', 'educ', 'income']]
y = data.loc[:, ['vote']]

oneHot = preprocessing.OneHotEncoder()
oneHot.fit(X)
X = oneHot.transform(X).toarray()
oneHot.fit(y)
y = oneHot.transform(y).toarray()

train_x,test_x, train_y, test_y = model_selection.train_test_split(X, y, test_size = 0.1, random_state=0)


n_samples = train_x.shape[0]

x = tf.placeholder(tf.float32, [None,train_x.shape[1]])
y = tf.placeholder(tf.float32,[None,2])


W = tf.Variable(np.zeros((train_x.shape[1], 2)),tf.float32,name="W")
b = tf.Variable(0.,dtype = tf.float32)


predicted_y1 = tf.add(tf.matmul(x,tf.cast(W,tf.float32) ), b)
print (predicted_y1.shape)
predicted_y = tf.nn.softmax(predicted_y1)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predicted_y, labels = y))


optimizer = tf.train.AdamOptimizer(0.0003).minimize(cross_entropy)

s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())

for i in range(40000):
    _, loss_i = s.run([optimizer,cross_entropy], {x: train_x, y: train_y})
    print("loss at iter %i:%.4f" % (i, loss_i))

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predicted_y,1), tf.argmax(y,1)), "float"))
accuracy_value = s.run(accuracy, feed_dict={x:test_x, y:test_y})
print (accuracy_value)
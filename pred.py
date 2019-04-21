import tensorflow as tf
import numpy as np
from utils import load_data, folder, cat2text, tcat2text
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
IMG_WIDTH = 100
IMG_HEIGHT = 30
x_data, y_data = load_data(folder)

x_train, x_test, y_train, y_test = train_test_split(np.array(x_data), np.array(y_data), test_size=0.2, random_state=30)

# 看看长啥样
# plt.imshow(x_test[0], cmap=plt.cm.binary)
# plt.show()

# x_train = x_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)/255.0
x_test = x_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)/255.0


model = tf.keras.models.load_model('cat2.h5')

pred = model.predict(x_test)


# 对比错了几个
for i, c in enumerate(y_test):
    if cat2text(y_test[i]) != tcat2text(pred[i]):
        print(cat2text(y_test[i]), tcat2text(pred[i]))



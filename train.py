import tensorflow as tf
from tensorflow import keras
import numpy as np
# from PIL import Image
from sklearn.model_selection import train_test_split

from utils import load_data, folder

IMG_WIDTH = 100
IMG_HEIGHT = 30
x_data, y_data = load_data(folder)

x_train, x_test, y_train, y_test = train_test_split(np.array(x_data), np.array(y_data), test_size=0.2, random_state=30)

x_train = x_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)/255.0
x_test = x_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)/255.0

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = keras.Sequential()
# 第一层卷积
model.add(keras.layers.Conv2D(256, (3, 3), input_shape=x_train.shape[1:]))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# # 第二层卷积
model.add(keras.layers.Conv2D(256, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# # 第三层卷积
model.add(keras.layers.Conv2D(128, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# # 拍平
model.add(keras.layers.Flatten())       # 一维
model.add(keras.layers.Dense(1024))     # 全连接层
model.add(keras.layers.Activation('relu'))  # 激活

model.add(keras.layers.Dense(248))    # 全连接层2
model.add(keras.layers.Activation('sigmoid'))   # softmax

# 编译
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 查看模型
model.summary()

# 开始
model.fit(x_train, y_train, epochs=10, batch_size=64)

# 跑测试集
val_loss, val_acc = model.evaluate(x_test, y_test)
print("测试集loss:", val_loss)
print("测试集acc:", val_acc)
model.save("cat2.h5")
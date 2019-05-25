import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers

from utils import load_data, TRAIN_DIR, LEN_GEN_CAT, LEN_CAT, IMG_HEIGHT, IMG_WIDTH, model_path, model_path
import time

NAME = "{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
x_data, y_data = load_data(TRAIN_DIR)

x_train, x_test, y_train, y_test = train_test_split(np.array(x_data), np.array(y_data), test_size=0.2, random_state=30)

x_train = x_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1) / 255.0
x_test = x_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1) / 255.0

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


def cnn(xTrain):
    """
    CNN 训练
    :param xTrain:
    :param yTrain:
    :return:
    """
    model = Sequential()
    # 第一层卷积
    model.add(Conv2D(64, (3, 3), input_shape=xTrain.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # # 第二层卷积
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # # 第三层卷积
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # # 拍平
    model.add(Flatten())  # 一维
    model.add(Dense(1024))  # 全连接层
    model.add(Activation('relu'))  # 激活

    model.add(Dense(LEN_GEN_CAT * LEN_CAT))  # 全连接层2
    model.add(Activation('sigmoid'))  # softmax < sigmoid

    # 编译
    # adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy',  # categorical_crossentropy  binary_crossentropy
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def main():
    """
    main
    :return:
    """

    model = cnn(x_train)

    # 查看模型
    model.summary()
    # 开始
    model. fit(x_train, y_train, epochs=10, batch_size=32, callbacks=[tensorboard])

    # 跑测试集
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print("测试集loss:", val_loss)
    print("测试集acc:", val_acc)
    model.save(model_path)


if __name__ == '__main__':
    main()

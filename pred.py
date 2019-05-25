import tensorflow as tf
import numpy as np
from utils import load_data, VAL_DIR, cat2text, IMG_WIDTH, IMG_HEIGHT, tcat2text
# from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt

x_data, y_data = load_data(VAL_DIR)


# 看看长啥样
# plt.imshow(x_data[0], cmap=plt.cm.binary)
# plt.show()

x_test = np.array(x_data).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1) / 255.0

# print(x_test.shape)

model = tf.keras.models.load_model('cat.h5')

pred = model.predict(x_test)

_sum = 0
for i, c in enumerate(y_data):
    if cat2text(y_data[i]) != cat2text(pred[i]):
        print(cat2text(y_data[i]), cat2text(pred[i]))
        _sum += 1

print("测试样本共", x_test.shape[0])
print("错误了 %s 个" % _sum)
print("正确概率为：%s" % ((x_test.shape[0] - _sum) / x_test.shape[0]))

import os
import numpy as np
from PIL import Image
import time
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

str_num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
str_up_let = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']
str_up = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y', 'z']

# 配置区
GEN_CAT = str_num + str_up_let + str_up
# GEN_CAT = str_num
LEN_GEN_CAT = len(GEN_CAT)  # 将会出现的字符总数
LEN_CAT = 4  # 验证码长度
# train_folder = "./verification"
TRAIN_DIR = "./train"
VAL_DIR = "./val"
model_path = "./cat.h5"
IMG_HEIGHT = 60
IMG_WIDTH = 100


def pos2idx(pos):
    """
    位置信息转字符
    :param pos:
    :return:
    """
    if pos < 10:
        char_code = pos + ord('0')
    elif pos < 36:
        char_code = pos - 10 + ord('A')
    elif pos < 62:
        char_code = pos - 36 + ord('a')
    return chr(char_code)


def idx2pos(idx):
    """
    字符转换成位置信息
    :param idx:
    :return:
    """
    k = ord(idx) - 48  # 因为ASCII表中  48是字符 0
    if k > 9:
        k = ord(idx) - 55
        if k > 35:
            k = ord(idx) - 61
    return k


def text2cat(text):
    cat = np.zeros(LEN_CAT * LEN_GEN_CAT)
    for i, c in enumerate(text):
        idx = i * LEN_GEN_CAT + idx2pos(c)
        cat[idx] = 1
    return cat


def cat2text(cat):
    """
    位置信息转字符信息
    :param cat: 位置信息
    :return:
    """
    text = []
    for i in range(0, LEN_CAT):
        text.append(pos2idx(np.argmax(cat[i * LEN_GEN_CAT: i * LEN_GEN_CAT + LEN_GEN_CAT])))

    return "".join(text)


def tcat2text(cat):
    """
    位置信息转字符信息
    :param cat: 位置信息
    :return:
    """
    str1 = []
    str2 = []
    str3 = []
    str4 = []
    # y_test
    for i, c in enumerate(cat):
        # print(i)
        if i < 62:
            str1.append(c)
        elif 62 <= i < 124:
            str2.append(c)
        elif 124 <= i < 186:
            str3.append(c)
        else:
            str4.append(c)

    print(str1)
    print(str2)
    print(str3)
    print(str4)
    s1 = pos2idx(np.argmax(str1))
    s2 = pos2idx(np.argmax(str2))
    s3 = pos2idx(np.argmax(str3))
    s4 = pos2idx(np.argmax(str4))

    return "{}{}{}{}".format(s1, s2, s3, s4)


def read_img(img_name):
    """
    图片二值化
    :param img:
    :return:
    """
    img = Image.open(img_name).convert('L')
    data = np.array(img)
    return data


def load_data(folder_dir):
    """
    加载数据
    :param folder:
    :return:
    """
    # img = np.zeros([IMG_HEIGHT * IMG_WIDTH])
    img = []
    label = []
    for img_path in os.listdir(folder_dir):
        label.append(text2cat(img_path.split("_")[0]))
        fd = os.path.join(folder_dir, img_path)
        image = read_img(fd)
        img.append(image)
        # img[:] = image.flatten() / 255
    return img, label


def mondif(folder_dir):
    for img_path in os.listdir(folder_dir):
        # print(f"{folder_dir}/{img_path}")
        new_name = img_path.split(".")[0]
        if not new_name.find("_"):
            continue
        os.rename(f"{folder_dir}/{img_path}", f"{folder_dir}/{new_name}_1.png")


def data_increase(folder_dir):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    for img_path in os.listdir(folder_dir):
        # print(f"{folder_dir}/{img_path}")
        new_name = img_path.split(".")[0]
        # os.rename(f"{folder_dir}/{img_path}", f"{folder_dir}/{new_name}_1.png")
        img = load_img(f'{folder_dir}/{img_path}')  # 这是一个PIL图像
        x = img_to_array(img)  # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)
        x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)

        # 下面是生产图片的代码
        # 生产的所有图片保存在 `preview/` 目录下
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=TRAIN_DIR, save_prefix=new_name, save_format='png'):
            i += 1
            if i > 3:
                break  # 否则生成器会退出循环



if __name__ == '__main__':
    # X, Y = load_data(folder)
    # print(X.shape)
    # print(Y[0])
    # print(X.shape)
    # pos = text2cat('0Ad1')
    # print(pos)

    # text = cat2text(pos)
    # print(text)
    # mondif(TRAIN_DIR)
    # mondif(VAL_DIR)
    # data_increase(TRAIN_DIR)
    pass

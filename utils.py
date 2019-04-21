import os
import numpy as np
from PIL import Image

str_num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
str_up_let = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']
str_up = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y', 'z']

GEN_CAT = str_num + str_up_let + str_up
LEN_GEN_CAT = len(GEN_CAT)
CAT2CHA = dict(zip(range(LEN_GEN_CAT), GEN_CAT))
CHA2CAT = dict(zip(GEN_CAT, range(LEN_GEN_CAT)))
LEN_CAT = 4  # 验证码长度
folder = ".\\verification"
IMG_HEIGHT = 30
IMG_WIDTH = 100


# print(CAT2CHA)
# print(CHA2CAT)


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
    s1 = pos2idx(np.argmax(str1))
    s2 = pos2idx(np.argmax(str2))
    s3 = pos2idx(np.argmax(str3))
    s4 = pos2idx(np.argmax(str4))

    return "{}{}{}{}".format(s1, s2, s3, s4)


def cat2text(cat):
    """
    位置信息转字符信息
    :param cat: 位置信息
    :return:
    """
    cat_pos = cat.nonzero()[0]
    # print(cat_pos)
    text = []
    for i, c in enumerate(cat_pos):
        char_index = c % 62
        char_code = pos2idx(char_index)
        text.append(char_code)
    return "".join(text)


def read_img(img_name):
    """
    图片二值化
    :param img:
    :return:
    """
    img = Image.open(img_name).convert('L')
    data = np.array(img)
    return data


def convert2gray(img):
    """
    把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
    :param img:
    :return:
    """
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def load_data(folders):
    """
    加载数据
    :param folder:
    :return:
    """
    # img = np.zeros([IMG_HEIGHT * IMG_WIDTH])
    img = []
    label = []
    for img_path in os.listdir(folders):
        label.append(text2cat(img_path.split(".")[0]))
        fd = os.path.join(folder, img_path)
        image = read_img(fd)
        img.append(image)
        # img[:] = image.flatten() / 255
    return img, label


if __name__ == '__main__':
    # X, Y = load_data(folder)
    # print(X.shape)
    # print(Y[0])
    # print(X.shape)
    # pos = text2cat('0Ad1')
    # print(pos)

    # text = cat2text(pos)
    # print(text)
    pass

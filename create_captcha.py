# pip install captcha
from captcha.image import ImageCaptcha
from utils import IMG_WIDTH, IMG_HEIGHT, TRAIN_DIR, VAL_DIR
import os
import random


def create():
    file_name = ""
    for i in range(4):
        char = random.choice([chr(random.randint(65, 90)),
                              str(random.randint(0, 9))])
        file_name += char
    return file_name


def clean_same_captcha():
    dirs = [TRAIN_DIR, VAL_DIR]
    train_file_list = []
    train_path_list = []
    val_file_list = []
    val_path_list = []
    clean_list = []
    for d in dirs:

        for img_path in os.listdir(d):
            if d == TRAIN_DIR:
                filename = img_path.split("_")[0]
                train_file_list.append(filename)
                train_path_list.append(img_path)
            else:
                filename = img_path.split("_")[0]
                val_file_list.append(filename)
                val_path_list.append(img_path)

    len_val = len(val_file_list)
    for val in range(len_val):
        if val_file_list[val] in train_file_list:
            clean_list.append(val_path_list[val])

    for cl in clean_list:
        print(f"{VAL_DIR}/{cl}")
        os.remove(f"{VAL_DIR}/{cl}")
        # break


def main():
    dirs = [TRAIN_DIR, VAL_DIR]
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

        if d == TRAIN_DIR:
            step = 10000
        else:
            step = 1000

        for ns in range(step):
            file_name = create()
            img = ImageCaptcha(width=IMG_WIDTH, height=IMG_HEIGHT)
            image = img.generate_image(file_name)
            image.save(f'{d}/{file_name}_1.png')


if __name__ == '__main__':
    main()
    clean_same_captcha()

from PIL import Image, ImageDraw, ImageFont
import random
from utils import IMG_WIDTH, IMG_HEIGHT, TRAIN_DIR, VAL_DIR
import os
import time

def main():
    create_dirs = [TRAIN_DIR, VAL_DIR]

    for fdir in create_dirs:
        if not os.path.exists(fdir):
            os.mkdir(fdir)

        if fdir == TRAIN_DIR:
            step = 11000
        else:
            step = 1100

        for ns in range(step):
            img = Image.new(mode="RGB", size=(IMG_WIDTH, IMG_HEIGHT), color=(255, 255, 255))

            draw = ImageDraw.Draw(img, mode="RGB")

            font = ImageFont.truetype("C:\\Windows\\Fonts\\Arial.ttf", 28)

            file_name = ""
            for i in range(4):
                char = random.choice([chr(random.randint(65, 90)), chr(random.randint(97, 122)),
                                      str(random.randint(0, 9))])

                color = (random.randint(0, 255), random.randint(
                    0, 255), random.randint(0, 255))
                file_name += char
                draw.text([i * 24, 0], char, color, font=font)

            # img.show()

            with open(f'{fdir}/{file_name}_{int(time.time())}.png', "wb") as f:
                img.save(f, format="png")


if __name__ == '__main__':
    main()

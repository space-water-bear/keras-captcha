from PIL import Image, ImageDraw, ImageFont
import random

for ns in range(10000):
    img = Image.new(mode="RGB", size=(100, 30), color=(255, 255, 255))

    draw = ImageDraw.Draw(img, mode="RGB")

    font = ImageFont.truetype("C:\\Windows\\Fonts\\Arial.ttf", 28)

    file_name = ""
    for i in range(4):
        char = random.choice([chr(random.randint(65, 90)),
                              str(random.randint(0, 9))])

        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        file_name += char
        draw.text([i * 24, 0], char, color, font=font)

    # img.show()

    with open(".\\verification\\%s.png" % file_name, "wb") as f:
        img.save(f, format="png")

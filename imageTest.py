import os

import cv2
import numpy as np

from libs.testGen import generateTestSquare
from libs.imageFusion import TransparentImageOverlay


# set image creation runs
count = 5

# Create test folder
myPath = f"output/test/run-X{count}-V"
prefixNbr = 1
folder = myPath+f"{prefixNbr}/"

while True:
    if not os.path.isdir(folder):
        print("folder doesn't exist", folder)
        os.makedirs(folder)
        break

    # check if folder maybe empty
    if len(os.listdir(folder)) == 0:
        print("folder is empty -> use it", folder)
        break

    prefixNbr = prefixNbr + 1
    folder = myPath+f"{prefixNbr}/"


print("created folder", folder)


# generate random transformed rectangles for test purposes
testEngine = generateTestSquare(count, folder)
images = testEngine.test_run()

print(f"Generating {count} images...")
imageAnalyzer = TransparentImageOverlay("", "")
print("...successfull")

def hughLineCheck(contour, background):
    lines = imageAnalyzer.get_huffman_lines(contour)

    if lines is None:
        print("LineCheck: No contour was found on the image")
        return

    print(f"LineCheck: found {len(lines)} through Hough-Transformation")

    for line in lines:
        x1,y1,x2,y2 = imageAnalyzer.polar_to_cart(line)
        cv2.line(background, (x1, y1), (x2, y2), (0, 0, 255), 4)

    cv2.imwrite(folder + f"{x}-polar.png", background)
    print("LineCheck: Saved report image")


blue_c = (255, 0, 0)
red_c = (0,0,255)


def hughPointLinecheck(contour, background):
    cv2.imwrite(folder + f"{x}-canny.png", contour)

    lines, mapped_lines = imageAnalyzer.get_huffman_pointLines(contour)
    mapped_background = background.copy()

    if lines is not None:
        print(f"PointCheck: found {len(lines)} lines in Image")

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(background, (x1, y1), (x2, y2), blue_c, 3)
            cv2.circle(background, (x1,y1), 15, blue_c, 3)
            cv2.circle(background, (x2,y2), 15, blue_c, 3)
    else:
        print("PointCheck: No contour was found on the image")
        return False

    cv2.imwrite(folder + f"{x}-coord.png", background)


    if mapped_lines is not None:
        mapped_count = len(mapped_lines)
        print(f"PointCheck: mapped to {mapped_count} lines in Image")

        for line in mapped_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mapped_background, (x1, y1), (x2, y2), red_c, 3)
            cv2.circle(mapped_background, (x1,y1), 15, red_c, 3)
            cv2.circle(mapped_background, (x2,y2), 15, red_c, 3)

        cv2.imwrite(folder + f"{x}-mapped.png", mapped_background)
        print("PointCheck: Saved report image")

        if mapped_count == 4:
            return True

    else:
        print("PointCheck: No contour was found on the image")
        return False



successCount = 0
for (x,data) in enumerate(images):
    # try to find solution
    print(f"\nAnalyze image {x + 1}/{count}")

    canny = data.get("canny")
    bg = data.get("original")

    #hughLineCheck(canny, bg.copy())
    if hughPointLinecheck(canny, bg):
        successCount += 1

print(f"\n\nAnalysed with Rate of {successCount} / {count}")

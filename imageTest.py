import cv2
import numpy as np

from libs.testGen import generateTestSquare
from libs.imageFusion import TransparentImageOverlay


myPath = "output/test/"
count = 100

testEngine = generateTestSquare(count, myPath)
images = testEngine.test_run()

print(f"Generating {count} images...")
imageAnalyzer = TransparentImageOverlay("", "")
print("...successfull")


for (x,data) in enumerate(images):
    # try to find solution
    canny = data.get("canny")
    bg = data.get("original")

    print(f"\nAnalyze image {x+1}/{count}")
    lines = imageAnalyzer.get_huffman_lines(canny)

    if lines is None:
        print("No contour was found on the image")
        continue

    print(f"found {len(lines)} through Hough-Transformation")

    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(bg, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(myPath+f"{x}-x.png", bg)
    print("Saved report image")


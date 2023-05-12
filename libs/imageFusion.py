import cv2
import numpy as np
import os
import math
from math import pi, cos, sin

class TransparentImageOverlay:
    def __init__(self, bottom_image_path, top_image_path, points):
        self.bottom_image_path = bottom_image_path
        self.top_image_path = top_image_path

        self.transformPoints = np.array(points, dtype=np.float32)

    def overlay_images(self, output_path, keepScreenshot=False):

        # Öffne die beiden Bilder und konvertiere sie in BGRA-Modus, um Transparenz zu unterstützen
        background = cv2.imread(self.bottom_image_path, cv2.IMREAD_UNCHANGED)
        h, w = background.shape[:2]
        bg_size = (w, h)

        # Calculate 4-Point transformation
        self.getScreenPoints(background)
        exit()

        # Oeffne Screenshot
        screenshot = cv2.imread(self.top_image_path, cv2.IMREAD_UNCHANGED)
        alpha_screen, screen_glare = self.process_green_pixels(background)

        # Transform and Mask Screenshot
        transformed_screen = self.add_four_point_transform(screenshot, bg_size, alpha_screen)
        screen_glow = self.create_border_glow(transformed_screen, bg_size)

        layer1 = self.overlay(screen_glow, background)
        layer2 = self.overlay(transformed_screen, layer1)
        layer3 = self.overlay(screen_glare, layer2)

        # Save the new image in BGRA format
        cv2.imwrite(output_path, layer3)

        if keepScreenshot is False:
            self.removeScreenshotTemp()

    def getScreenPoints(self, img):
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define a range of green color in HSV
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 255, 255])

        # Threshold the image to get only green pixels with high intensity
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]

        # Apply the mask to the input image
        green_image = cv2.bitwise_and(img, img, mask=mask)

        # Convert the masked image to grayscale, blur it, and find edges
        gray = cv2.cvtColor(green_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        #cv2.imwrite("output/countour.png", edged)

        contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lines = np.array(self.find_edge_coordinates(contours))
        collinear_lines = self.find_collinear_lines(lines)[:4]

        if len(collinear_lines) < 4:
            print("Could not find any screen")
            return

        for line in collinear_lines:
            p1, p2 = line
            cv2.line(img, p1, p2, (255, 0, 0, 255), 5)  # draw line in green color with thickness of 2

        # Draw the rectangle on the image
        cv2.imwrite("output/test.png", img)


    def find_edge_coordinates(self, contours):
        # Iterate over the contours and find the rectangle with rounded corners
        lines = []

        for cnt in contours:
            # Approximate the contour with a polygon

            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.1 * perimeter, True)
#
            # Check if the polygon has four sides
            if len(approx) == 4:
                print("This contour has roughly 4 sides")

                # Reduce Point count
                somePoints = cnt[::2]
                for ix, val in enumerate(somePoints):
                    if ix >= len(somePoints) - 1:
                        break

                    current_point = somePoints[ix][0]
                    next_point = somePoints[ix+1][0]

                    lines.append((current_point, next_point))
                break
        return lines

    def find_collinear_lines(self, lineArr):
        edges = []

        def line_slope(line):
            epsilon = 1e-6
            (x1, y1), (x2, y2) = line

            if x2-x1 < epsilon:
                slope = 999
            else:
                slope = (y2 - y1) / (x2 - x1)

            return slope

        line_builder = []
        for ix, line in enumerate(lineArr):
            if ix >= len(lineArr) - 1:
                break

            current_line = lineArr[ix]
            next_line = lineArr[ix+1]

            current_slope = line_slope(current_line)
            next_slope = line_slope(next_line)

            if current_slope == next_slope:

                if len(line_builder) < 1:
                    # start new Line segment
                    line_builder = [current_line[0], next_line[1]]
                else:
                    # add line end to segment
                    line_builder[1] = next_line[1]

            else:
                if len(line_builder) > 1:
                    # previous line was collinear, but this one isn't anymore
                    edges.append(line_builder)
                    line_builder = []

        print(edges)
        return edges

    def getBiggest(self, lineArray):
        def line_length(line):
            (x1, y1), (x2, y2) = line
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        biggestLines = sorted(lineArray, key=line_length, reverse=True)[:4]

        return biggestLines


    def process_green_pixels(self, input_image):
        # TODO: Nur innerhalb der Screen-Punkte suchen
        # Bisher wird das ganze Bild abgesucht

        print("...generate screen alpha-channel")
        height, width, channels = input_image.shape

        # Initialize empty images for the alpha mask and the overlay layer
        alpha_mask = np.zeros((height, width), dtype=np.uint8)
        overlay_layer = np.zeros((height, width, 4), dtype=np.uint8)

        # Go over each pixel in the input image
        for y in range(height):
            for x in range(width):
                b, g, r, a = input_image[y, x]

                # Check if the pixel has a high green value and low red and blue values
                if g > 150 and r < 150 and b < 150:
                    # Create a masking pixel on the alpha mask
                    alpha_mask[y, x] = 255

                    # Add a pixel onto the overlay layer with transparency
                    overlay_layer[y, x] = (b, int((r + b) / 2), r, (255 - g))

        # grow mask by 1px
        kernel = np.ones((3, 3), np.uint8)
        alpha_mask = cv2.dilate(alpha_mask, kernel, iterations=1)

        return alpha_mask, overlay_layer

    def create_border_glow(self, screen, size, darkThreshold = 0.2):

        # Erstelle ein neues transparentes leeres Bild auf Größe des Hintergrunds
        empty_image = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        image = self.overlay(screen, empty_image)

        # blur new generated image violently
        blurred = cv2.blur(image, (100, 100))
        def bezier_curve(t, sludge):
            p1 = [0,sludge]
            p2 = [1 - sludge, 1]
            return ((1 - t) ** 3 * p1[0]) + (3 * (1 - t) ** 2 * t * p1[1]) + (3 * (1 - t) * t ** 2 * p2[0]) + (t ** 3 * p2[1])

        def adjust_alpha(pixel):
            color_value = np.mean(pixel[:3]) / 255
            color_value = bezier_curve(color_value, sludge=0.6)
            pixel[3] = color_value * 255
            return pixel

        # Apply the alpha adjustment
        for i in range(blurred.shape[0]):
            for j in range(blurred.shape[1]):
                blurred[i, j] = adjust_alpha(blurred[i, j])

        return blurred

    def add_four_point_transform(self, screenshot, size, mask):
        print("...transforming Screenshot")
        # Calculate the 4-point transformation
        pts_src = np.array(
            [[0, 0], [screenshot.shape[1], 0], [screenshot.shape[1], screenshot.shape[0]], [0, screenshot.shape[0]]],
            dtype=np.float32)

        h, status = cv2.findHomography(pts_src, self.transformPoints)
        out = cv2.warpPerspective(screenshot, h, size)

        # Apply a Mask to the wraped image
        if len(mask) > 0:
            out = cv2.merge((out[:, :, 0], out[:, :, 1], out[:, :, 2], mask))
            print("...applying screen masking")

        return out

    def overlay(self, image1, image2):
        print("...blending")
        # Add the two images together, preserving alpha channel
        alpha_screenshot = image1[:, :, 3] / 255.0
        alpha_background = image2[:, :, 3] / 255.0
        alpha_combined = alpha_screenshot + (1 - alpha_screenshot) * alpha_background
        final_alpha = (alpha_combined * 255).astype(np.uint8)

        for c in range(0, 3):
            image2[:, :, c] = np.divide(
                alpha_screenshot * image1[:, :, c] + (1 - alpha_screenshot) * image2[:, :, c] * alpha_background,
                alpha_combined,
                out=np.zeros_like(alpha_combined),
                where=(alpha_combined != 0)
            )

        image2[:, :, 3] = final_alpha

        return image2

    def removeScreenshotTemp(self):
        os.remove(self.top_image_path)

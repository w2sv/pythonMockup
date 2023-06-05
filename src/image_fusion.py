import math
import os

import cv2
import numpy as np

from src.cli import bcolors


class TransparentImageOverlayer:
    def __init__(self, bottom_image_path, top_image_path):
        self.bottom_image_path = bottom_image_path
        self.top_image_path = top_image_path

        self.transformPoints = []
        self.perimterToleranz = 0.05

    def overlay_images(self, folder, file, keepScreenshot=False):

        # Öffne die beiden Bilder und konvertiere sie in BGRA-Modus, um Transparenz zu unterstützen
        background = cv2.imread(self.bottom_image_path, cv2.IMREAD_UNCHANGED)

        # Calculate 4-Point transformation
        self.transformPoints = np.array(self.getScreenPoints(background, folder, file), dtype='uint8')

        for x in self.transformPoints:
            print(x)

        # Set Background, Glow, (Screen + Mask) and Screen-Glare  into one composition
        imageComp = self.applyLayer(background)

        # Save the new image
        cv2.imwrite(folder + file + ".png", imageComp)

        if keepScreenshot is False:
            self.removeScreenshotTemp()

    def applyLayer(self, bg, easy_mode=False):
        h, w = bg.shape[:2]
        bg_size = (w, h)

        # Oeffne Screenshot
        if not os.path.isfile(self.top_image_path):
            raise Exception(f"{bcolors.FAIL}Could not find screenshot file on drive.{bcolors.ENDC}")
        screenshot = cv2.imread(self.top_image_path, cv2.IMREAD_UNCHANGED)

        # Get Mask from Green Pixels for Screenshot masking, calculate glare from green offset
        alpha_screen, screen_glare = self.process_green_pixels(bg)

        # Transform and Mask Screenshot
        transformed_screen = self.add_four_point_transform(screenshot, bg_size, alpha_screen)

        if not easy_mode:
            screen_glow = self.create_border_glow(transformed_screen, bg_size)

            print(
                f"...blending layer1: {bcolors.OKBLUE}background{bcolors.ENDC} and {bcolors.OKBLUE}screen-glow{bcolors.ENDC}")
            layer1 = self.overlay(screen_glow, bg)
            print(
                f"...blending layer2: {bcolors.OKBLUE}layer1{bcolors.ENDC} and {bcolors.OKBLUE}screenshot{bcolors.ENDC}")
            layer2 = self.overlay(transformed_screen, layer1)
            print(
                f"...blending final composition: {bcolors.OKBLUE}layer2{bcolors.ENDC} and {bcolors.OKBLUE}screen-glare{bcolors.ENDC}")
            return self.overlay(screen_glare, layer2)
        print(f"...blending layer")
        return self.overlay(transformed_screen, bg)

    def getScreenPoints(self, img, dir, file):
        print(f"{bcolors.OKCYAN}calculation screen position on mockup:{bcolors.ENDC}")
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define a range of green color in HSV
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 255, 255])

        # Threshold the image to get only green pixels with high intensity
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
        #cv2.imwrite(dir+".temp/"+file+"-mask.png", mask)

        # Apply the mask to the input image
        green_image = cv2.bitwise_and(img, img, mask=mask)

        # Convert the masked image to grayscale, blur it, and find edges
        gray = cv2.cvtColor(green_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
<<<<<<< HEAD:src/image_fusion.py
        edged = cv2.Canny(gray, 75, 200)
        # cv2.imwrite(dir+".temp/"+file+"-contour.png", edged)
=======
        edged = cv2.Canny(gray, 50, 150)
>>>>>>> 730c7b8 (first hugh-transformation attempts):libs/imageFusion.py

        dilated = cv2.dilate(edged, None)
        cv2.imwrite(dir+".temp/"+file+"-contour.png", dilated)

        # Huff-Transformation Huff-Lines

        lines = cv2.HoughLines(
            image=dilated,
            rho= 0.01,   # max_pixel_to_line_distance
            theta= np.pi / 180, # degree_tolerancee
            threshold= 50  # min_weights_threshold
        )

        if lines is None:
            raise Exception("Could not find any Lines with Hugh-Transform")

        demoImage = img.copy()
<<<<<<< HEAD:src/image_fusion.py
        for line in lines:
            point, p2 = line
            cv2.circle(demoImage, point, 5, (255, 0, 0, 255), -1)
        cv2.imwrite(dir + ".temp/" + file + "-border.png", demoImage)
=======
        # Draw Lines
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(demoImage, (x1, y1), (x2, y2), (0, 0, 255,255), 3)
>>>>>>> 730c7b8 (first hugh-transformation attempts):libs/imageFusion.py

        cv2.imwrite(dir+".temp/"+file+"-all-lines.png", demoImage)

<<<<<<< HEAD:src/image_fusion.py
        for line in collinear_lines:
            p1, p2 = line
            cv2.line(demoImage, p1, p2, (0, 0, 150, 255), 5)
        cv2.imwrite(dir + ".temp/" + file + "-border.png", demoImage)
=======
        def line_intersection(line1, line2):
            rho1, theta1 = line1[0]
            rho2, theta2 = line2[0]
            if np.isclose(theta1, theta2):  # the lines are parallel
                return None
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
>>>>>>> 730c7b8 (first hugh-transformation attempts):libs/imageFusion.py

            return [x0, y0]

        intersections = []
        for i, line1 in enumerate(lines):
            for line2 in lines[i + 1:]:
                intersection = line_intersection(line1, line2)
                if intersection is not None and intersection[0] > 0 and intersection[1] > 0:
                    intersections.append(intersection)

        print(intersections)

<<<<<<< HEAD:src/image_fusion.py
        if len(border_lines) > 1:
            for lineF in border_lines:
                p1, p2 = lineF
                cv2.circle(demoImage, p1, 20, (255, 255, 255, 255), 3)
                cv2.circle(demoImage, p2, 20, (255, 255, 255, 255), 3)
                cv2.line(demoImage, p1, p2, (0, 0, 255, 255), 3)

            cv2.imwrite(dir + ".temp/" + file + "-border.png", demoImage)

        if len(border_lines) < 4:
            raise Exception(f"{bcolors.FAIL}Could not find any screen{bcolors.ENDC}")

        # calculate intersection points from four lines
        screen_coordinates = self.get_line_intersection(border_lines)

        if len(screen_coordinates) > 1:
            for pt in screen_coordinates:
                cv2.circle(demoImage, pt, 10, (0, 0, 255, 255), 8)
            cv2.imwrite(dir + ".temp/" + file + "-border.png", demoImage)

        if len(screen_coordinates) != 4:
            raise Exception(f"{bcolors.FAIL}Could not calculate 4-Point transformation from data{bcolors.ENDC}")

        # draw Demo Image
        return screen_coordinates

    def contour_to_lines(self, contours):
        # Iterate over the contours and find the rectangle with rounded corners
        lines = []

        for cnt in contours:
            # Approximate the contour with a polygon
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.perimterToleranz * perimeter, True)
            #
            # Check if the polygon has four sides
            if len(approx) == 4:
                print(f"{bcolors.OKGREEN}Screen-contour found on mockup{bcolors.ENDC}")

                # Reduce Point count for better results
                for ix, val in enumerate(cnt):
                    if ix >= len(cnt) - 1:
                        break

                    current_point = cnt[ix][0]
                    next_point = cnt[ix + 1][0]

                    lines.append((current_point, next_point))
                break

        return lines

    def line_to_vector(self, l1, normalize=False):
        vector = (
            l1[1][0] - l1[0][0],
            l1[1][1] - l1[0][1]
        )

        if normalize:
            magnitude = math.sqrt(sum(component ** 2 for component in vector))
            return tuple(component / magnitude for component in vector)

        return vector

    def find_collinear_lines(self, lineArr):

        def cross_product(v1, v2):
            crossProduct = (v1[0] * v2[1]) - (v1[1] * v2[0])
            return abs(crossProduct)

        edges = []

        line_builder = ()
        last_crossProduct = 0

        for ix, line in enumerate(lineArr):
            if ix >= len(lineArr) - 1:
                break

            current_line = line
            next_line = lineArr[ix + 1]

            emptyLine = len(line_builder) < 1

            # Make Vektors from Line Segments
            if emptyLine:
                v1, v2 = self.line_to_vector(current_line), self.line_to_vector(next_line)
            else:
                v1, v2 = self.line_to_vector(line_builder), self.line_to_vector(next_line)

            crossProduct = cross_product(v1, v2)
            crossDifference = abs(crossProduct - last_crossProduct)
            last_crossProduct = crossProduct

            if crossDifference < self.crossToleranz:
                # cross Product is small
                if emptyLine:
                    # start new Line segment
                    line_builder = (current_line[0], next_line[1])
                else:
                    # add line end to segment
                    line_builder = (line_builder[0], next_line[1])
            else:
                # cross Product is > 0
                if not emptyLine:
                    # previous line was collinear, but this one isn't anymore
                    edges.append(line_builder)
                    line_builder = ()

        return edges

    def get_line_intersection(self, lines):
        def line_equation(line):
            (x1, y1), (x2, y2) = line
            m = (y2 - y1) / (x2 - x1) if x1 != x2 else float('inf')
            b = y1 - m * x1
            return m, b

        def intersection(line1, line2):
            m1, b1 = line_equation(line1)
            m2, b2 = line_equation(line2)
            if m1 == m2:  # Parallel lines, no intersection
                return None
            if m1 == float('inf'):  # Line1 is vertical
                x = line1[0][0]
                y = m2 * x + b2
            elif m2 == float('inf'):  # Line2 is vertical
                x = line2[0][0]
                y = m1 * x + b1
            else:
                x = (b2 - b1) / (m1 - m2)
                y = m1 * x + b1

            # Only return the point if it's within the desired range
            if 0 <= x <= 5000 and 0 <= y <= 5000:
                return x, y
            else:
                return None

        def angle(point):
            x, y = point
            return math.atan2(y, x)

        def distance_to_origin(point):
            x, y = point
            return math.sqrt(x ** 2 + y ** 2)

        intersections = []
        for i in range(4):
            for j in range(i + 1, 4):
                point = intersection(lines[i], lines[j])
                if point is not None:
                    intersections.append(point)

        intersections = sorted(intersections, key=distance_to_origin)[:4]
        origin_min_distance_point = min(intersections, key=distance_to_origin)
        intersections = sorted(intersections, key=angle)
        intersections.remove(origin_min_distance_point)
        intersections = [origin_min_distance_point] + intersections

        return [(int(x), int(y)) for x, y in intersections]

    def get_biggest(self, lineArray):
        def length(line):
            (x1, y1), (x2, y2) = line
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        lineArray.sort(key=length, reverse=True)
        bigger = lineArray[:4]
        return bigger
=======
        if len(intersections) != 4:
            raise Exception(f"{bcolors.FAIL}Could not calculate 4-Point transformation from data{bcolors.ENDC}")

        # draw Demo Image
        return intersections
>>>>>>> 730c7b8 (first hugh-transformation attempts):libs/imageFusion.py

    def process_green_pixels(self, input_image):
        # TODO: Nur innerhalb der Screen-Punkte suchen
        # Bisher wird das ganze Bild abgesucht

        print(f"{bcolors.OKCYAN}Generate screen alpha-channel{bcolors.ENDC}")
        height, width, channels = input_image.shape

        # Initialize empty images for the alpha mask and the overlay layer
        alpha_mask = np.zeros((height, width), dtype=np.uint8)
        overlay_layer = np.zeros((height, width, 4), dtype=np.uint8)

        # Go over each pixel in the input image
        for y in range(height):
            for x in range(width):
                b, g, r, a = input_image[y, x]

                # Check if the pixel has a high green value and low red and blue values
                if g > 150 > r and b < 150:
                    # Create a masking pixel on the alpha mask
                    alpha_mask[y, x] = 255

                    # Add a pixel onto the overlay layer with transparency
                    overlay_layer[y, x] = (b, int((r + b) / 2), r, (255 - g))

        # grow mask by 1px
        kernel = np.ones((3, 3), np.uint8)
        alpha_mask = cv2.dilate(alpha_mask, kernel, iterations=1)

        return alpha_mask, overlay_layer

    def create_border_glow(self, screen, size):
        print(f"{bcolors.OKCYAN}calculating screen glow{bcolors.ENDC}")
        # Erstelle ein neues transparentes leeres Bild auf Größe des Hintergrunds
        empty_image = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        image = self.overlay(screen, empty_image)

        # blur new generated image violently
        print("...blurr image")
        blurred = cv2.blur(image, (100, 100))

        def bezier_curve(t, sludge):
            p1 = [0, sludge]
            p2 = [1 - sludge, 1]
            return ((1 - t) ** 3 * p1[0]) + (3 * (1 - t) ** 2 * t * p1[1]) + (3 * (1 - t) * t ** 2 * p2[0]) + (
                    t ** 3 * p2[1])

        def adjust_alpha(pixel):
            color_value = np.mean(pixel[:3]) / 255
            color_value = bezier_curve(color_value, sludge=0.6)
            pixel[3] = color_value * 255
            return pixel

        # Apply the alpha adjustment
        print("...adjust blurr opacity based on brightness")
        for i in range(blurred.shape[0]):
            for j in range(blurred.shape[1]):
                blurred[i, j] = adjust_alpha(blurred[i, j])

        return blurred

    def add_four_point_transform(self, screenshot, size, mask):
        print(f"{bcolors.OKCYAN}transforming screenshot{bcolors.ENDC}")
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

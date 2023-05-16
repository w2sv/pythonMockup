import cv2
import numpy as np
import os
import math
from libs.bColor import bcolors


class TransparentImageOverlay:
    def __init__(self, bottom_image_path, top_image_path):
        self.bottom_image_path = bottom_image_path
        self.top_image_path = top_image_path

        self.transformPoints = []
        self.perimterToleranz = 0.05


    def overlay_images(self, folder, file, keepScreenshot=False):

        # Öffne die beiden Bilder und konvertiere sie in BGRA-Modus, um Transparenz zu unterstützen
        background = cv2.imread(self.bottom_image_path, cv2.IMREAD_UNCHANGED)

        # Calculate 4-Point transformation
        contourImage = self.get_contour(background)
        cv2.imwrite(dir+".temp/"+file+"-contour.png", contourImage)

        # get Huffman Lines
        contourLines = self.get_huffman_lines(contourImage)

        intersections = self.lines_to_intersection(contourLines)
        self.transformPoints = np.array(intersections, dtype='uint8')

        # Set Background, Glow, (Screen + Mask) and Screen-Glare  into one composition
        imageComp = self.applyLayer(background)

        # Save the new image
        cv2.imwrite(folder+file+".png", imageComp)

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

            print(f"...blending layer1: {bcolors.OKBLUE}background{bcolors.ENDC} and {bcolors.OKBLUE}screen-glow{bcolors.ENDC}")
            layer1 = self.overlay(screen_glow, bg)
            print(f"...blending layer2: {bcolors.OKBLUE}layer1{bcolors.ENDC} and {bcolors.OKBLUE}screenshot{bcolors.ENDC}")
            layer2 = self.overlay(transformed_screen, layer1)
            print(f"...blending final composition: {bcolors.OKBLUE}layer2{bcolors.ENDC} and {bcolors.OKBLUE}screen-glare{bcolors.ENDC}")
            return self.overlay(screen_glare, layer2)
        print(f"...blending layer")
        return self.overlay(transformed_screen, bg)

    def get_contour(self, img):
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
        edged = cv2.Canny(gray, 50, 150, apertureSize=3)
        # make edge contour little bit bigger
        dilated = cv2.dilate(edged, None)

        return dilated

    def get_huffman_lines(self, contour):
        # Huff-Transformation Huff-Lines
        lines = cv2.HoughLines(
            image=contour,
            rho=.1,  # max_pixel_to_line_distance
            theta=np.pi / 180,  # 5*360 steps
            threshold= 150  # min_weights_threshold
        )

        return lines

    def lines_to_intersection(self, lines):

        if lines is None:
            raise Exception("Could not find any Lines with Hugh-Transform")

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

            return [x0, y0]

        intersections = []
        for i, line1 in enumerate(lines):
            for line2 in lines[i + 1:]:
                intersection = line_intersection(line1, line2)
                if intersection is not None and intersection[0] > 0 and intersection[1] > 0:
                    intersections.append(intersection)

        if len(intersections) != 4:
            raise Exception(f"{bcolors.FAIL}Could not calculate 4-Point transformation from data{bcolors.ENDC}")

        # draw Demo Image
        return intersections

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
                if g > 150 and r < 150 and b < 150:
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
            p1 = [0,sludge]
            p2 = [1 - sludge, 1]
            return ((1 - t) ** 3 * p1[0]) + (3 * (1 - t) ** 2 * t * p1[1]) + (3 * (1 - t) * t ** 2 * p2[0]) + (t ** 3 * p2[1])

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

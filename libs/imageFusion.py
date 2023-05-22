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
        self.size = (0,0)

    def overlay_images(self, folder, file, keepScreenshot=False):

        # Öffne die beiden Bilder und konvertiere sie in BGRA-Modus, um Transparenz zu unterstützen
        background = cv2.imread(self.bottom_image_path, cv2.IMREAD_UNCHANGED)

        h, w = background.shape[:2]
        self.size = (w, h)

        # Calculate 4-Point transformation
        contourImage = self.get_contour(background)
        cv2.imwrite(dir+".temp/"+file+"-contour.png", contourImage)

        # get Huffman Lines
        contourLines = self.get_huffman_lines(contourImage)

        # intersections = self.lines_to_intersection(contourLines)
        # self.transformPoints = np.array(intersections, dtype='uint8')

        # Set Background, Glow, (Screen + Mask) and Screen-Glare  into one composition
        imageComp = self.applyLayer(background)

        # Save the new image
        cv2.imwrite(folder+file+".png", imageComp)

        if keepScreenshot is False:
            self.removeScreenshotTemp()

    def applyLayer(self, bg, easy_mode=False):

        # Oeffne Screenshot
        if not os.path.isfile(self.top_image_path):
            raise Exception(f"{bcolors.FAIL}Could not find screenshot file on drive.{bcolors.ENDC}")
        screenshot = cv2.imread(self.top_image_path, cv2.IMREAD_UNCHANGED)

        # Get Mask from Green Pixels for Screenshot masking, calculate glare from green offset
        alpha_screen, screen_glare = self.process_green_pixels(bg)

        # Transform and Mask Screenshot
        transformed_screen = self.add_four_point_transform(screenshot, alpha_screen)

        if not easy_mode:
            screen_glow = self.create_border_glow(transformed_screen)

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
        h, w = contour.shape[:2]
        self.size = (w, h)

        # Huff-Transformation Huff-Lines in polar form
        return cv2.HoughLines(
            image=contour,
            rho=1,  # max_pixel_to_line_distance
            theta=np.pi / 180,  # 5*360 steps
            threshold=150  # min_weights_threshold
        )

    def get_huffman_pointLines(self, contour):
        h, w = contour.shape[:2]
        self.size = (w, h)
        # Huff-Transformation Huff-Lines in Point (X, Y) form

        def line_length(line):
            x1, y1, x2, y2 = line[0]
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        h, w = contour.shape[:2]
        min_line_length = min([w, h]) // 4

        lines = cv2.HoughLinesP(
            image=contour,
            rho= 1,
            theta= np.pi / 720,
            threshold= 5,
            minLineLength=min_line_length,
            maxLineGap=None
        )

        return lines

    import numpy as np

    def combine_overlapping_lines(self, lines):
        def get_line_degree(degree_line):
            _, degree_theta = self.cart_to_polar(degree_line[0])
            degree_out = int((math.degrees(degree_theta) + 360) % 180) + 1
            return degree_out

        # Sort lines by degree
        sorted_degree_lines = sorted(lines, key=lambda l: get_line_degree(l))

        # check which one is horizontal / vertical
        x1,y1,x2,y2 = sorted_degree_lines[0][0]
        first_line_horizontal = abs(x2-x1) > abs(y2-y1)

        # Assosiate degree change with lines
        last_degree = None
        change_degree_assoc = []
        for idx, vec in enumerate(sorted_degree_lines):
            # set last line if undefined -> for next loop
            curr_line_degree = get_line_degree(vec)
            if last_degree is None:
                last_degree = curr_line_degree
                continue

            # calculate difference to prev degree
            change = abs(curr_line_degree / last_degree)
            change_degree_assoc.append(change)
            # print(f"{curr_line_degree}°", int(change))
            last_degree = curr_line_degree

        # get Index, where to split between horizontal, vertical
        max_value = max(change_degree_assoc, key=lambda x: x)
        indices = [index for index, item in enumerate(change_degree_assoc) if item == max_value]
        print(f"Max change rate= {max_value} as Position {indices}")

        split_index = indices[0] +1
        group_one = sorted_degree_lines[:split_index] if first_line_horizontal else sorted_degree_lines[split_index:]
        group_two = sorted_degree_lines[split_index:] if first_line_horizontal else sorted_degree_lines[:split_index]

        print(f"{bcolors.OKGREEN}found {len(group_one)}-horizontal and {len(group_two)}-vertical{bcolors.ENDC}")

        return group_one, group_two

    def get_relevant_lines(self, line_group, is_horizontal):
        w, h = self.size
        line_skew = 5
        line_offset = 50
        vertical_line = ((w//2) - line_skew, line_offset, (w//2) + line_skew, h-line_offset)
        horizontal_line = (line_offset, (h//2) - line_skew, w-line_offset, (h//2) + line_skew)
        reference_line = vertical_line if is_horizontal else horizontal_line

        intersection_array = []
        for line in line_group:
            # calculate X/Y-Value for Intersection with reference line
            line_intersections = self.line_intersection(reference_line, line[0])
            if line_intersections is None:
                print(f"{bcolors.FAIL}No Intersection between Refline and line?{bcolors.ENDC}")
                print(f"Vertical Line and L: {line[0]}")
                continue
            else:
                # Add intersection Value (x or y) with line to list
                entry = (line_intersections, line)
                #print(f"{bcolors.OKGREEN}Found Intersection between Refline and line:{bcolors.ENDC}")
                #print(f"Horizontal Line and L: {line[0]}")
                intersection_array.append(entry)

        # sort Array for X / Y Value of intersection point
        sorted_data = sorted(intersection_array, key=lambda tup: tup[0][0 if is_horizontal else 1])

        return sorted_data, reference_line

        # return just first and last line
        # return sorted_data[0][1], sorted_data[-1][1]

    def polar_to_cart(self, line):
        r, theta = line
        x1 = r * np.cos(theta)
        y1 = r * np.sin(theta)
        x2 = x1 + 1000 * np.cos(theta + (np.pi / 2))
        y2 = y1 + 1000 * np.sin(theta + (np.pi / 2))
        return int(x1), int(y1), int(x2), int(y2)

    def cart_to_polar(self, line):
        x1, y1, x2, y2 = line

        dx = int(x2 - x1)
        dy = int(y2 - y1)

        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan2(dy, dx)

        return r, theta

    def line_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Calculate the intersection point of two infinite lines
        try:
            xy_factor = np.subtract(
                np.multiply(x3, y4),
                np.multiply(y3, x4)
            )
            xy_subtract = np.subtract(
                np.multiply(x1, y2),
                np.multiply(y1, x2)
            )
            x_num = np.subtract(
                np.multiply(xy_subtract, np.subtract(x3, x4)),
                np.multiply(np.subtract(x1, x2), xy_factor)
            )
            y_num = np.subtract(
                np.multiply(xy_subtract, np.subtract(y3, y4)),
                np.multiply(np.subtract(y1, y2), xy_factor)
            )
            denom = np.subtract(
                np.multiply(x1 - x2, y3 - y4),
                np.multiply(y1 - y2, x3 - x4)
            )

            if np.isclose(denom, 0, atol=1e-6):
                # Lines are parallel or coincident
                return None
            else:
                x = int(np.divide(x_num, denom))
                y = int(np.divide(y_num, denom))

                # Only return plausible values
                if np.logical_and(
                        np.logical_and(0 < x, x < self.size[0]),
                        np.logical_and(0 < y, y < self.size[1])
                ):
                    return x, y
                else:
                    return None
        except Exception as e:
            print(f"{bcolors.FAIL}Overflow in Intersection calculation: {e}{bcolors.ENDC}")
            return None

    def get_screen_position(self, line_group):
        intersection_points = []
        run_length = len(line_group)
        # Iterate through all combinations of lines
        for i in range(run_length):
            curr_line = line_group[i][0]
            next_line = line_group[i+1][0] if i+1 < run_length else line_group[0][0]

            print(f"check {curr_line} with {next_line}")

            intersect = self.line_intersection(curr_line, next_line)
            if intersect is not None:
                intersection_points.append(intersect)

        if len(intersection_points) == 4:
            return intersection_points
        else:
            return None

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

    def create_border_glow(self, screen):
        print(f"{bcolors.OKCYAN}calculating screen glow{bcolors.ENDC}")
        # Erstelle ein neues transparentes leeres Bild auf Größe des Hintergrunds
        empty_image = np.zeros((self.size[1], self.size[0], 4), dtype=np.uint8)
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

    def add_four_point_transform(self, screenshot, mask):
        print(f"{bcolors.OKCYAN}transforming screenshot{bcolors.ENDC}")
        # Calculate the 4-point transformation
        pts_src = np.array(
            [[0, 0], [screenshot.shape[1], 0], [screenshot.shape[1], screenshot.shape[0]], [0, screenshot.shape[0]]],
            dtype=np.float32)

        h, status = cv2.findHomography(pts_src, self.transformPoints)
        out = cv2.warpPerspective(screenshot, h, self.size)

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

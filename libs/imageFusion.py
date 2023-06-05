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
        self.size = (0, 0)

    def overlay_images(self, folder, file, keepScreenshot=False, debug=True):

        def print_lines(lines_data, image, color=(255,0,0)):
            if lines_data is not None:
                for lin in lines_data:
                    x1, y1, x2, y2 = lin[0]
                    cv2.line(image, (x1, y1), (x2, y2), color, 3)
                    cv2.circle(image, (x1, y1), 15, color, 3)
                    cv2.circle(image, (x2, y2), 15, color, 3)

        def print_dots(dot_data, image, color=(255,0,0)):
            if dot_data is not None:
                for dot in dot_data:
                    x, y, = dot
                    cv2.circle(image, (x, y), 15, color, 3)

        # Oeffne das Bild (rbga)
        background = cv2.imread(self.bottom_image_path, cv2.IMREAD_UNCHANGED)

        h, w = background.shape[:2]
        self.size = (w, h)

        # Calculate 4-Point transformation
<<<<<<< HEAD
        contour_image = self.get_contour(background)
        if debug:
            cv2.imwrite(folder+".temp/"+file+"-1-contour.png", contour_image)

        # get Hugh Lines
        all_lines = self.get_hough_lines(contour_image)
        print(f"{bcolors.OKBLUE}Hough-transform detected {len(all_lines)} lines in contour{bcolors.ENDC}")
        mapped_x, mapped_y = self.combine_overlapping_lines(all_lines)
=======
        contourImage = self.get_contour(background)
        cv2.imwrite(dir+".temp/"+file+"-contour.png", contourImage)

        # get Huffman Lines
        contourLines = self.get_huffman_lines(contourImage)

        intersections = self.lines_to_intersection(contourLines)
        self.transformPoints = np.array(intersections, dtype='uint8')
>>>>>>> 77979e8 (demo working)

        # debug Image
        if debug:
            all_lines_img = background.copy()
            print_lines(mapped_x, all_lines_img, (255,0,0, 255))
            print_lines(mapped_y, all_lines_img, (255,0,255, 255))
            cv2.imwrite(folder+".temp/"+file+"-2-houghLines.png", all_lines_img)

        sorted_y_intersect, vert_line = self.get_relevant_lines(mapped_x, True)
        sorted_x_intersect, hori_line = self.get_relevant_lines(mapped_y, False)
        if len(sorted_y_intersect) < 2 or len(sorted_x_intersect) < 2:
            print(f"{bcolors.FAIL}Could not find two lines per orientation{bcolors.ENDC}")
            return False

        y_points, y_lines = zip(*sorted_y_intersect)
        x_points, x_lines = zip(*sorted_x_intersect)
        four_borders = (y_lines[0], x_lines[-1], y_lines[-1], x_lines[0])

        if debug:
            all_intersection_img = background.copy()
            print_dots(x_points, all_intersection_img, (255,0,0, 255))
            print_lines([(hori_line,)], all_intersection_img, (255,0,0, 255))
            print_dots(y_points, all_intersection_img, (255,0,255, 255))
            print_lines([(vert_line,)], all_intersection_img,  (255,0,255, 255))
            cv2.imwrite(folder+".temp/"+file+"-3-lineCalculation.png", all_intersection_img)

        intersections = self.get_screen_position(four_borders)
        if intersections is None:
            print(f"{bcolors.FAIL}Could not find screen corners from data{bcolors.ENDC}")
            return False

        print(f"{bcolors.OKGREEN}four screen-corner coordinates calculated{bcolors.ENDC}")

        if debug:
            edged_img = background.copy()
            print_dots(intersections, edged_img, (255,255,255, 255))
            cv2.imwrite(folder+".temp/"+file+"-4-finalPoints.png", edged_img)

        self.transformPoints = np.array(intersections, dtype=np.float32)
        # Set Background, Glow, (Screen + Mask) and Screen-Glare  into one composition
        easy_mode = False
        print(f"{bcolors.OKCYAN}Starting image processing ...{bcolors.ENDC}")

        easy_compositing = "just screenshot position on mockup"
        hard_compositing = "background mockup image, screenshot, glow and glare"
        print(f"{bcolors.BOLD}Compositing {easy_compositing if easy_mode else hard_compositing}{bcolors.ENDC}")
        image_comp = self.applyLayer(background, easy_mode)

        # Save the new image
        cv2.imwrite(folder+file+".png", image_comp)

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

        # Apply the mask to the input image
        green_image = cv2.bitwise_and(img, img, mask=mask)

        # Convert the masked image to grayscale, blur it, and find edges
        gray = cv2.cvtColor(green_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 50, 150, apertureSize=3)
        # make edge contour little bit bigger
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edged, kernel)

        return dilated

<<<<<<< HEAD

    def get_hough_lines(self, contour):
        h, w = contour.shape[:2]
        self.size = (w, h)
        # Hugh-Transformation Hugh-Lines in Point (X, Y) form

        h, w = contour.shape[:2]
        min_line_length = min([w, h]) // 10

        lines = cv2.HoughLinesP(
            image=contour,
            rho=1,
            theta=np.pi / 720,
            threshold=50,
            minLineLength=min_line_length,
            maxLineGap=None
        )

        return lines

    def combine_overlapping_lines(self, lines):
        def get_line_degree(degree_line):
            x1, y1, x2, y2 = degree_line[0]
            slope_rad = math.atan2((y2 - y1), (x2 - x1))
            # Convert slope to degrees
            slope_deg = math.degrees(slope_rad)
            # If slope is negative, add 180 to it to make it positive
            if slope_deg < 0:
                slope_deg += 180

            # If slope is more than 90 degrees, subtract from 180 to make it less than or equal to 90
            if slope_deg > 90:
                slope_deg = 180 - slope_deg

            return int(slope_deg)

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
            change = abs(curr_line_degree - last_degree)
            change_degree_assoc.append(change)
            # print(f"{curr_line_degree}°", change)
            last_degree = curr_line_degree

        # get Index, where to split between horizontal, vertical
        max_value = max(change_degree_assoc, key=lambda x: x)
        indices = [index for index, item in enumerate(change_degree_assoc) if item == max_value]
        #print(f"Split Vertical to Horizontal at index {indices} after delta degree {max_value}")

        split_index = indices[0] +1
        group_one = sorted_degree_lines[:split_index] if first_line_horizontal else sorted_degree_lines[split_index:]
        group_two = sorted_degree_lines[split_index:] if first_line_horizontal else sorted_degree_lines[:split_index]

        print(f"{bcolors.OKGREEN}seperated {len(group_one)}-horizontal and {len(group_two)}-vertical{bcolors.ENDC} lines")

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
                print(f"{bcolors.FAIL}No Intersection between refline and line?{bcolors.ENDC}")
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

    def line_intersection(self, line1, line2):
        scaling_factor = max(*self.size)
        line1 = [coord / scaling_factor for coord in line1]
        line2 = [coord / scaling_factor for coord in line2]

        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        try:
            denom = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
            if abs(denom) < 1e-10:  # Consider lines parallel if denom is very small
                return None  # Lines are parallel

            xy_factor = (x3 * y4) - (y3 * x4)
            xy_subtract = (x1 * y2) - (y1 * x2)
            x_num = (xy_subtract * (x3 - x4)) - ((x1 - x2) * xy_factor)
            y_num = (xy_subtract * (y3 - y4)) - ((y1 - y2) * xy_factor)

            x = (x_num / denom) * scaling_factor
            y = (y_num / denom) * scaling_factor

            if -1 < x < self.size[0] and -1 < y < self.size[1]:
                return int(x), int(y)
            else:
=======
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
>>>>>>> 77979e8 (demo working)
                return None

        except Exception as e:
            print(f"{bcolors.FAIL}Overflow in Intersection calculation: {e}{bcolors.ENDC}")
            return None

    def get_screen_position(self, line_group):
        intersection_points = []
        num_lines = len(line_group)
        # Iterate through all combinations of lines
        for i in range(num_lines):
            curr_line = line_group[i][0]
            next_line = line_group[(i + 1) % num_lines][0]

<<<<<<< HEAD
            # print(f"check {curr_line} with {next_line}")
            intersect = self.line_intersection(curr_line, next_line)

            if intersect is not None:
                intersection_points.append(intersect)
            else:
                return None
=======
        if len(intersections) != 4:
            raise Exception(f"{bcolors.FAIL}Could not calculate 4-Point transformation from data{bcolors.ENDC}")
>>>>>>> 77979e8 (demo working)

        if len(intersection_points) == 4:
            # sort points in right order
            sorted_intersects = self.sort_points_clockwise(intersection_points)
            return sorted_intersects
        else:
            return None
    def sort_points_clockwise(self, points):
        # Function to calculate the Euclidean distance from a point to the origin
        def distance_to_origin(point):
            return math.sqrt(point[0] ** 2 + point[1] ** 2)

<<<<<<< HEAD
        # Function to calculate the angle a point forms with the positive x-axis
        def angle_to_x_axis(point):
            return math.atan2(-point[1], point[0])

        # Find the point closest to the origin
        start_point = min(points, key=distance_to_origin)

        # Sort the points in clockwise order, starting with the start_point
        sorted_points = sorted(points, key=angle_to_x_axis, reverse=True)

        # Make sure the start_point is first in the sorted list
        sorted_points.remove(start_point)
        sorted_points.insert(0, start_point)

        return sorted_points

=======
>>>>>>> 77979e8 (demo working)
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
        print("...blur image")
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
        print("...adjust blur opacity based on brightness")
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

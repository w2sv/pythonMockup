import random

import cv2
import numpy as np

class generateTestSquare:

    def __init__(self, count, path = "output/test/"):
        self.outputPath = path
        self.count = count

    def test_run(self):
        img_arr = []
        for x in range(self.count):
            contour, reference = self.generateDemoCanny(np.random.randint(1500, 3000), np.random.randint(1500,3000))
            #cv2.imwrite(f"{self.outputPath}{x}.png", newTestImage)

            img_arr.append({
                "canny": contour,
                "original": reference
            })

        return img_arr

    def generateDemoCanny(self, width, height):
        # Create a black background image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        white_ref = np.ones((height, width, 3), dtype=np.uint8) * 155

        r_pos, r_size, r_contour = self.create_round_rectangle(width, height)

        if r_pos[0] % 6 == 0:
            # Sometimes create a Notch onto r_contour
            n_pos, n_size, n_contour = self.create_round_rectangle(*r_size)
            r_contour[n_pos[1]:n_pos[1] + n_size[1], n_pos[0]:n_pos[0] + n_size[0]] = n_contour

        # perspectevly move two coordinates of the screen-edge
        transform_coordinates = self.randomTransform(
            top_left=r_pos,
            bottom_right=(r_pos[0]+r_size[0], r_pos[1]+r_size[1]),
            max = min(r_size) // 5
        )

        transform_screen = self.fourWayTransform(r_contour, (width, height), transform_coordinates)

        canny = cv2.cvtColor(self.overlay(image, transform_screen), cv2.COLOR_BGR2GRAY)
        reference = self.overlay(white_ref, transform_screen)

        return canny, reference

    def fourWayTransform(self, screen, bg_size, fourPointDst):
        # Define the four corner points in the source image
        # Calculate the 4-point transformation
        pts_src = np.array([
            [0, 0],[screen.shape[1], 0],
            [screen.shape[1], screen.shape[0]],
            [0, screen.shape[0]]
        ], dtype=np.float32)

        h, status = cv2.findHomography(pts_src, np.array([*fourPointDst], dtype=np.float32))
        return cv2.warpPerspective(screen, h, bg_size)

    def overlay(self, image1, image2):
        for c in range(0, 3):
            image2[:, :, c] = (image1[:, :, c] + image2[:, :, c])

        return image2

    def randomTransform(self, top_left, bottom_right, max):

        opposite_edge = bool(random.getrandbits(1))
        x1_trans, y1_trans, x2_trans, y2_trans = np.random.randint(-max, max, size=4)

        top_left = top_left if opposite_edge else (top_left[0] + x1_trans, top_left[1] + y1_trans)
        top_right = (bottom_right[0] + x1_trans, top_left[1] + y1_trans) if opposite_edge else (bottom_right[0], top_left[1])
        bottom_right = bottom_right if opposite_edge else (bottom_right[0] + x2_trans, bottom_right[1] + y2_trans)
        bottom_left = (top_left[0] - x2_trans, bottom_right[1] - y2_trans) if opposite_edge else (top_left[0], bottom_right[1])

        return top_left, top_right, bottom_right, bottom_left

    def create_round_rectangle(self, width, height):

        # Generate random 4-Point Rectangle
        rect_width = np.random.randint(0.5 * width, 0.8 * width)
        x_move_dist = width - rect_width
        rect_x = np.random.randint(0.2*x_move_dist, 0.8*x_move_dist)

        rect_height = np.random.randint(0.5 * height, 0.8 * height)
        y_move_dist = height - rect_height
        rect_y = np.random.randint(0.2*y_move_dist, 0.8*y_move_dist)

        # Create black backdrop
        screen_contour = np.zeros((rect_height, rect_width, 3), dtype=np.uint8)
        #screen_contour[:, :, 2] = 150

        thickness = 5
        b = thickness+1  # border

        color = (255, 255, 255)

        devide = np.random.randint(10, 100)
        radius = 0 if devide % 7 == 0 else min((rect_width, rect_height)) // devide

        # top line
        p1, p2 = ((radius + b, b), (rect_width - radius - b, b))
        cv2.line(screen_contour, p1, p2, color, thickness)
        # top-right corner
        cv2.ellipse(screen_contour, (p2[0], p2[1] + radius), (radius, radius), 270, 0, 90, color, thickness)

        # right line
        p1, p2 = ((rect_width - b, radius + b), (rect_width - b, rect_height - radius - b))
        cv2.line(screen_contour, p1, p2, color, thickness)
        # bottom-right corner
        cv2.ellipse(screen_contour, (p2[0] - radius, p2[1]), (radius, radius), 0, 0, 90, color, thickness)

        # bottom line
        p1, p2 = ((rect_width - radius - b, rect_height - b), (radius + b, rect_height - b))
        cv2.line(screen_contour, p1, p2, color, thickness)
        # bottom-left corner
        cv2.ellipse(screen_contour, (p2[0], p2[1] - radius), (radius, radius), 90, 0, 90, color, thickness)

        # left line
        p1, p2 = ((b, rect_height - radius - b), (b, radius + b))
        cv2.line(screen_contour, p1, p2, color, thickness)
        # top-left corner
        cv2.ellipse(screen_contour, (p2[0] + radius, p2[1]), (radius, radius), 180, 0, 90, color, thickness)

        return (rect_x, rect_y), (rect_width, rect_height), screen_contour







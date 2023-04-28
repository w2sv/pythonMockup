import cv2
import numpy as np
import os

class TransparentImageOverlay:
    def __init__(self, bottom_image_path, top_image_path, points):
        self.bottom_image_path = bottom_image_path
        self.top_image_path = top_image_path
        self.transformPoints = np.array(points, dtype=np.float32)

    def overlay_images(self, output_path, keepScreenshot=False):

        # Öffne die beiden Bilder und konvertiere sie in BGRA-Modus, um Transparenz zu unterstützen
        background = cv2.imread(self.bottom_image_path, cv2.IMREAD_UNCHANGED)

        # Oeffne Screenshot
        screenshot = cv2.imread(self.top_image_path, cv2.IMREAD_UNCHANGED)
        alpha_screen, screen_glare = self.process_green_pixels(background)

        # Transform and Mask Screenshot
        transformed_screen = self.add_four_point_transform(screenshot, background.shape, alpha_screen)
        screen_glow = self.create_border_glow(transformed_screen, background.shape)

        layer1 = self.overlay(screen_glow, background)
        layer2 = self.overlay(transformed_screen, layer1)
        layer3 = self.overlay(screen_glare, layer2)

        # Save the new image in BGRA format
        cv2.imwrite(output_path, layer3)

        if keepScreenshot is False:
            self.removeScreenshotTemp()

    def process_green_pixels(self, input_image):
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

        #grow mask by 1px
        kernel = np.ones((3, 3), np.uint8)
        alpha_mask = cv2.dilate(alpha_mask, kernel, iterations=1)

        return alpha_mask, overlay_layer


    def create_border_glow(self, screen, bg_shape, glow_radius= 500, glow_intensity=15):
        h, w = bg_shape[:2]

        # Erstelle ein neues transparentes leeres Bild auf Größe des Hintergrunds
        empty_image = np.zeros((h, w, 4), dtype=np.uint8)
        image = self.overlay(screen, empty_image)

        border_mask = self.detect_borders(image)
        border_mask = border_mask.astype(np.float32) / 255.0

        glow = np.zeros((h, w, 4), dtype=np.float32)
        glow[:, :, :3] = image[:, :, :3].astype(np.float32) / 255.0
        glow[:, :, 3] = border_mask

        ksize = 2 * (glow_radius // 2) + 1
        blurred_glow = cv2.GaussianBlur(glow, (ksize, ksize), 0)

        img_float = image.astype(np.float32) / 255.0
        intensified_glow = blurred_glow * glow_intensity

        alpha_glow = intensified_glow[:, :, 3]
        alpha_image = img_float[:, :, 3]
        alpha_combined = alpha_glow + (1 - alpha_glow) * alpha_image

        output = np.copy(img_float)
        for c in range(3):
            output[:, :, c] = np.divide(
                alpha_glow * intensified_glow[:, :, c] + (1 - alpha_glow) * img_float[:, :, c] * alpha_image,
                alpha_combined,
                out=np.zeros_like(alpha_combined),
                where=(alpha_combined != 0)
            )
        output[:, :, 3] = (1 - alpha_image) * alpha_glow

        return (output * 255.0).astype(np.uint8)

    def detect_borders(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        border_mask = np.zeros_like(thresholded)
        cv2.drawContours(border_mask, contours, -1, 255, 1)

        return border_mask

    def add_four_point_transform(self, screenshot, background_shape, mask):
        print("...transforming Screenshot")
        # Calculate the 4-point transformation
        pts_src = np.array(
            [[0, 0], [screenshot.shape[1], 0], [screenshot.shape[1], screenshot.shape[0]], [0, screenshot.shape[0]]],
            dtype=np.float32)

        h, status = cv2.findHomography(pts_src, self.transformPoints)
        out = cv2.warpPerspective(screenshot, h, (background_shape[1], background_shape[0]))

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
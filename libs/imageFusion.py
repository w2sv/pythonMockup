from PIL import Image, ImageDraw, ImageFilter
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
        screenshot = cv2.imread(self.top_image_path, cv2.IMREAD_UNCHANGED)

        # Füge das Leuchten (verwischter Hintergrund) hinter dem Screenshot hinzu
        screen_glow = self.add_blur(background, screenshot, 100)

        glowBackground = self.overlay(screen_glow, background);

        # Füge die 4-Punkt-Transformation hinzu
        final_image = self.add_four_point_transform(glowBackground, screenshot)

        # Save the new image in BGRA format
        cv2.imwrite(output_path, final_image)

        if keepScreenshot is not True:
            self.removeScreenshotTemp()

    def add_blur(self, background, screenshot, blur_radius=30):
        h, w = background.shape[:2]

        # Erstelle ein neues transparentes leeres Bild auf Größe des Hintergrunds
        empty_image = np.zeros((h, w, 4), dtype=np.uint8)

        # Setze den Screenshot mit 4-Point-Transform an die richtige Stelle auf das leere Bild
        empty_image = self.add_four_point_transform(empty_image, screenshot)

        # Wende einen Blur auf das komplette Bild an
        blurred_image = cv2.GaussianBlur(empty_image, (0,0), blur_radius)

        return cv2.addWeighted(blurred_image, 1.5, np.zeros(blurred_image.shape, blurred_image.dtype), 0, 1)

    def add_four_point_transform(self, background, screenshot):
        # Calculate the 4-point transformation
        pts_src = np.array(
            [[0, 0], [screenshot.shape[1], 0], [screenshot.shape[1], screenshot.shape[0]], [0, screenshot.shape[0]]],
            dtype=np.float32)

        h, status = cv2.findHomography(pts_src, self.transformPoints)
        out = cv2.warpPerspective(screenshot, h, (background.shape[1], background.shape[0]))

        # Add the transformed image onto the background image
        alpha_screenshot = out[:, :, 3] / 255.0
        alpha_background = background[:, :, 3] / 255.0
        alpha_combined = alpha_screenshot + (1 - alpha_screenshot) * alpha_background
        final_alpha = (alpha_combined * 255).astype(np.uint8)

        for c in range(0, 3):
            background[:, :, c] = np.divide(
                alpha_screenshot * out[:, :, c] + (1 - alpha_screenshot) * background[:, :, c] * alpha_background,
                alpha_combined,
                out=np.zeros_like(alpha_combined),
                where=(alpha_combined != 0)
            )

        background[:, :, 3] = final_alpha

        return background

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
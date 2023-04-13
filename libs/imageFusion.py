from PIL import Image, ImageDraw, ImageFilter
import os

class TransparentImageOverlay:
    def __init__(self, bottom_image_path, top_image_path, x, y, width):
        self.bottom_image_path = bottom_image_path
        self.top_image_path = top_image_path
        self.x = x
        self.y = y
        self.width = width

    def overlay_images(self, output_path, keepScreenshot=False):
        # Öffne die beiden Bilder und konvertiere sie in RGBA-Modus, um Transparenz zu unterstützen
        with Image.open(self.bottom_image_path) as b_image, Image.open(self.top_image_path) as t_image:
            background = b_image.convert('RGBA')
            screenshot = self.transformScreenshot(t_image).convert('RGBA')

            # Place white blur rectangle
            backgroundComp = self.add_blur(background, screenshot, 150, 0.75)

            # Einfügen des oberen Bildes an der gewünschten Position auf dem Hintergrund mit BlurBox
            backgroundComp.paste(screenshot, (self.x, self.y), mask=screenshot)

            # Speichern des neuen Bildes
            backgroundComp.save(output_path)

            if keepScreenshot is not True:
                self.removeScreenshotTemp()

    def transformScreenshot(self, image):
        # Berechne die Höhe des oberen Bildes basierend auf dem Seitenverhältnis und der Breite
        width_percent = (self.width / float(image.size[0]))
        height = int((float(image.size[1]) * float(width_percent)))

        # Skaliere das obere Bild auf die gewünschte Größe
        newScreenSize = image.resize((self.width, height))
        return newScreenSize

    def add_blur(self,background,screenshot,blur_radius=100, opacity=1.0):

        # Create a new image with the same size as the original image
        rectangle = Image.new("RGBA", background.size, (255, 255, 255, 0))

        # Draw the white rectangle
        draw = ImageDraw.Draw(rectangle)
        draw.rectangle([self.x, self.y, self.x + screenshot.width, self.y + screenshot.height], fill=(255, 255, 255, (int)(opacity * 255)))

        # Blur the edges of the rectangle
        blurBox = rectangle.filter(ImageFilter.GaussianBlur(blur_radius))

        # Place the blurred rectangle onto the original image
        result = Image.alpha_composite(background, blurBox)

        return result

    def removeScreenshotTemp(self):
        os.remove(self.top_image_path)
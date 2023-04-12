from PIL import Image, ImageFilter
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
            top_image = t_image.convert('RGBA')

            # Berechne die Höhe des oberen Bildes basierend auf dem Seitenverhältnis und der Breite
            width_percent = (self.width / float(top_image.size[0]))
            height = int((float(top_image.size[1]) * float(width_percent)))

            # Skaliere das obere Bild auf die gewünschte Größe
            top_image = top_image.resize((self.width, height))


            # Calculate the size of the mask with a 10% margin around the smaller image
            margin_width = top_image.width // 10
            margin_height = top_image.height // 10
            mask_width = top_image.width + margin_width * 2
            mask_height = top_image.height + margin_height * 2
            mask_size = (mask_width, mask_height)

            alpha = top_image.split()[-1]
            alpha_blur = alpha.filter(ImageFilter.GaussianBlur(radius=10))
            alpha_blur_large = alpha_blur.resize(mask_size, resample=Image.BOX)

            # Create a mask from the larger blurred alpha channel that matches the size of the smaller image
            maskImg = Image.new('RGBA', top_image.size, (0, 0, 0, 0))
            maskImg.paste(alpha_blur_large, (-margin_width, -margin_height))

            # Einfügen des Screen Glows
            background.paste(top_image, (self.x, self.y), mask=maskImg)

            # Einfügen des oberen Bildes an der gewünschten Position auf dem Hintergrund
            # background.paste(top_image, (self.x, self.y), mask=top_image)

            # Speichern des neuen Bildes
            background.save(output_path)

            if keepScreenshot is not True:
                self.removeScreenshotTemp()

    def removeScreenshotTemp(self):
        os.remove(self.top_image_path)
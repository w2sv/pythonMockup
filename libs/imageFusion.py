from PIL import Image
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
            bottom_image = b_image.convert('RGBA')
            top_image = t_image.convert('RGBA')

            # Berechne die Höhe des oberen Bildes basierend auf dem Seitenverhältnis und der Breite
            width_percent = (self.width / float(top_image.size[0]))
            height = int((float(top_image.size[1]) * float(width_percent)))

            # Skaliere das obere Bild auf die gewünschte Größe
            top_image = top_image.resize((self.width, height))

            # Erstellen eines transparenten Hintergrundbildes, das der Größe des unteren Bildes entspricht
            background = Image.new('RGBA', bottom_image.size, (255, 255, 255, 0))

            # Einfügen des unteren Bildes in den Hintergrund
            background.paste(bottom_image, (0, 0))

            # Einfügen des oberen Bildes an der gewünschten Position auf dem Hintergrund
            background.paste(top_image, (self.x, self.y), mask=top_image)

            # Speichern des neuen Bildes
            background.save(output_path)

            if keepScreenshot is not True:
                self.removeScreenshotTemp()

    def removeScreenshotTemp(self):
        os.remove(self.top_image_path)
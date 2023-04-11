from PIL import Image

class TransparentImageOverlay:

    def __init__(self, background_img_path, overlay_img_path, x, y, width):
        self.background_img_path = background_img_path
        self.overlay_img_path = overlay_img_path
        self.x = x
        self.y = y
        self.width = width

    def overlay_images(self, output_path):
        # Load the background and overlay images
        background_img = Image.open(self.background_img_path).convert("RGBA")
        overlay_img = Image.open(self.overlay_img_path).convert("RGBA")

        # Calculate the height of the overlay image based on its aspect ratio
        aspect_ratio = overlay_img.width / overlay_img.height
        height = round(self.width / aspect_ratio)

        # Resize the overlay image to the desired width and height
        resized_overlay_img = overlay_img.resize((self.width, height))

        # Create a new transparent image with the same dimensions as the background image
        new_img = Image.new("RGBA", background_img.size, (0, 0, 0, 0))

        # Paste the background image onto the new image
        new_img.paste(background_img, (0, 0))

        # Paste the resized overlay image onto the new image at the specified X/Y coordinates
        new_img.paste(resized_overlay_img, (self.x, self.y), mask=resized_overlay_img)

        # Save the new image as a PNG file
        new_img.save(output_path, format="PNG")

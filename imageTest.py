from libs.imageFusion import TransparentImageOverlay

newPath = "output/test/"

photoTest2 = TransparentImageOverlay(
    bottom_image_path="./src/mockup/xdr-display.png",
    top_image_path="output/democodespace/.temp/b00684e19beda813.png"
)
photoTest2.overlay_images(newPath, "xdr", keepScreenshot=True)

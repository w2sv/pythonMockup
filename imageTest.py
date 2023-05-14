from libs.imageFusion import TransparentImageOverlay

newPath = "output/tests/"

photoTest1 = TransparentImageOverlay(
    bottom_image_path="./src/mockup/macbook-air-m1.png",
    top_image_path="output/democodespace/.temp/8790c788911946d0.png"
)
photoTest1.overlay_images(newPath+"macbook", keepScreenshot=True)


photoTest2 = TransparentImageOverlay(
    bottom_image_path="./src/mockup/iphone-14-pro.png",
    top_image_path="output/democodespace/.temp/b00684e19beda813.png"
)
photoTest2.overlay_images(newPath+"iphone", keepScreenshot=True)

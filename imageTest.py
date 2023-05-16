from libs.imageFusion import TransparentImageOverlay

newPath = "output/test/"


photoTest2 = TransparentImageOverlay(
    bottom_image_path="./src/mockup/iphone-14-pro-01.png",
    #bottom_image_path="./src/mockup/macbook-air-m1-03.png",
    #bottom_image_path="./src/mockup/xdr-display-02.png",
    top_image_path="output/probierklavier/.temp/0b056899b89dd90b.png"
)
photoTest2.overlay_images(newPath, "demo", keepScreenshot=False)

from src.image_fusion import TransparentImageOverlay

newPath = "output/test/"


photoTest2 = TransparentImageOverlay(
    bottom_image_path="./assets/mockup/iphone-14-pro-02.png",
    #bottom_image_path="./src/mockup/macbook-air-m1-03.png",
    #bottom_image_path="./src/mockup/xdr-display-02.png",
    top_image_path="output/probierklavier/.temp/0b056899b89dd90b.png"
)
photoTest2.overlay_images(newPath, "demo", keepScreenshot=False)

from libs.Promt import PromtManager
from libs.selenium import WebsiteScreenshot
from libs.imageFusion import TransparentImageOverlay
import time as t

# Initilize information query
consoleInterface = PromtManager(1)

# Take Screenshot
mockup = consoleInterface.mockupDevice
screenshotSize = mockup.get("screen")
screenshot = WebsiteScreenshot(
    url=consoleInterface.url,
    width=screenshotSize.get("width"),
    height=screenshotSize.get("height"),
    cookieClass=consoleInterface.hideClass,
    waitTime=5
)

# Create new Overlay Image
print("mergin images...")
screenPos = mockup.get("position")
photoBooth = TransparentImageOverlay(
    bottom_image_path=mockup.get("mockupImage"),
    top_image_path=screenshot.tempPath,
    points=[
        screenPos["p1"],
        screenPos["p2"],
        screenPos["p3"],
        screenPos["p4"]
    ]
)

fileName = "".join(x for x in mockup.get("name") if x.isalnum())

imgName = f"output/{fileName}-{t.time()}.png";
photoBooth.overlay_images(imgName, keepScreenshot=False)
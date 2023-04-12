from libs.Promt import PromtManager
from libs.selenium import WebsiteScreenshot
from libs.imageFusion import TransparentImageOverlay

# Initilize information query
consoleInterface = PromtManager()

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
    x=screenPos.get("left"),
    y=screenPos.get("top"),
    width=screenPos.get("width")
)
photoBooth.overlay_images("output/newMockup.png", keepScreenshot=False)
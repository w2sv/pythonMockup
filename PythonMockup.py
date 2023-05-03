from libs.Promt import PromtManager
from libs.selenium import WebsiteScreenshot
from libs.imageFusion import TransparentImageOverlay
import time as t
from libs.bColor import bcolors


class PythonMockup:

    def __init__(self):
        # Initilize information query
        PromtManager(self.process)

    def process(self, selectedDevices, promterInstance):

        for x in range(len(selectedDevices)):
            mockup = selectedDevices[x]

            print(f"{bcolors.UNDERLINE}Device generating: {mockup.get('name')}{bcolors.ENDC}")

            screenshotSize = mockup.get("screen")
            screenshot = WebsiteScreenshot(
                url=promterInstance.url,
                width=screenshotSize.get("width"),
                height=screenshotSize.get("height"),
                cookieClass=promterInstance.hideClass,
                waitTime=5
            )

            # Create new Overlay Image
            print("Starting image processor")
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


PythonMockup()
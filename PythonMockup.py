import os

from libs.promt import PromtManager
from libs.selenium import WebsiteScreenshot
from libs.imageFusion import TransparentImageOverlay
import time as t
from libs.bColor import bcolors
import tldextract


class PythonMockup:

    def __init__(self):
        # Initilize information query
        PromtManager(self.process)

    def process(self, selectedDevices, promterInstance):

        # create new output folder
        webNameInfo = tldextract.extract(promterInstance.url)
        folderName = self.sanitze(webNameInfo.subdomain + webNameInfo.domain)
        newPath = f"output/{folderName}/"

        if not os.path.exists(newPath):
            os.makedirs(newPath)

        # start selenium Engine
        screenshotEngine = WebsiteScreenshot(
            url=promterInstance.url,
            directory=newPath + "temp/",
            cookieClass=promterInstance.hideClass,
            waitTime=5,
        )

        for x in range(len(selectedDevices)):
            mockup = selectedDevices[x]

            print(f"\n{bcolors.UNDERLINE}Device generating:{bcolors.ENDC} {bcolors.OKCYAN}{mockup.get('name')}{bcolors.ENDC}")

            screenshotSize = mockup.get("screen")
            newScreenshot = screenshotEngine.take_screenshot(
                screenshotSize.get("width"),
                screenshotSize.get("height")
            )

            # Create new Overlay Image
            print("Starting image processor")
            screenPos = mockup.get("position")
            photoBooth = TransparentImageOverlay(
                bottom_image_path=mockup.get("mockupImage"),
                top_image_path=newScreenshot,
                points=[
                    screenPos["p1"],
                    screenPos["p2"],
                    screenPos["p3"],
                    screenPos["p4"]
                ]
            )

            fileName = self.sanitze(mockup.get("name"))
            imgName = newPath + fileName + ".png"

            photoBooth.overlay_images(imgName, keepScreenshot=False)


        tempPath = "/".join(newScreenshot.split("/")[:-1]) + "/"
        os.rmdir(tempPath)
        print(f"Remove temp folder at {tempPath}")

        screenshotEngine.closeBrowser()

    def sanitze(self, string):
        return "".join(x for x in string if x.isalnum())


PythonMockup()
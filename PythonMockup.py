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
        folderName = self.sanitize(webNameInfo.subdomain + webNameInfo.domain)
        newPath = f"output/{folderName}/"

        if not os.path.exists(newPath):
            os.makedirs(newPath)

        # start selenium Engine
        screenshotEngine = WebsiteScreenshot(
            url=promterInstance.url,
            directory=newPath+".temp/",
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

            mockupArr = mockup.get("mockupImage")
            for ix,n in enumerate(mockupArr):
                if not os.path.isfile(n):
                    continue

                realIndx = ix+1
                print(f"\n{bcolors.HEADER}Generating {realIndx}/{len(mockupArr)} mockups for {mockup.get('name')}{bcolors.ENDC}")
                photoBooth = TransparentImageOverlay(
                    bottom_image_path=n,
                    top_image_path=newScreenshot
                )

                fileName = self.sanitize(mockup.get("name")) + f"-{(ix+1):02d}"

                photoBooth.overlay_images(newPath, fileName, keepScreenshot= (realIndx < len(mockupArr)))


        screenshotEngine.closeBrowser()

    def sanitize(self, string):
        return "".join(x for x in string if x.isalnum())


PythonMockup()
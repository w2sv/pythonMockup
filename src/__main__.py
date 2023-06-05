import os
import tldextract

from src.cli import bcolors
from src.image_fusion import TransparentImageOverlayer
from src.cli.prompt import PromptManager
from src.screenshot_engine import WebsiteScreenshotEngine


def process(selectedDevices, prompt_manager: PromptManager):

    # create new output folder
    webNameInfo = tldextract.extract(prompt_manager.url)
    folderName = sanitize(webNameInfo.subdomain + webNameInfo.domain)
    newPath = f"output/{folderName}/"

    if not os.path.exists(newPath):
        os.makedirs(newPath)

    # start selenium Engine
    screenshotEngine = WebsiteScreenshotEngine(
        url=prompt_manager.url,
        directory=newPath + ".temp/",
        cookieClass=prompt_manager.hideClass,
        waitTime=5,
    )

    for mockup in selectedDevices:
        print(
            f"\n{bcolors.UNDERLINE}Device generating:{bcolors.ENDC} {bcolors.OKCYAN}{mockup.get('name')}{bcolors.ENDC}")

        screenshotSize = mockup.get("screen")
        newScreenshot = screenshotEngine.take_screenshot(
            screenshotSize.get("width"),
            screenshotSize.get("height")
        )

        # Create new Overlay Image
        print("Starting image processor")

        mockupArr = mockup.get("mockupImage")
        for ix, n in enumerate(mockupArr):
            if not os.path.isfile(n):
                continue

            realIndx = ix + 1
            print(
                f"\n{bcolors.HEADER}Generating {realIndx}/{len(mockupArr)} mockups for {mockup.get('name')}{bcolors.ENDC}")
            photoBooth = TransparentImageOverlayer(
                bottom_image_path=n,
                top_image_path=newScreenshot
            )

            fileName = sanitize(mockup.get("name")) + f"-{(ix + 1):02d}"

            photoBooth.overlay_images(newPath, fileName, keepScreenshot=(realIndx < len(mockupArr)))

    screenshotEngine.closeBrowser()


def sanitize(string):
    return "".join(x for x in string if x.isalnum())


if __name__ == '__main__':
    PromptManager(process)
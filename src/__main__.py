import os
import tldextract

from src.cli import bcolors
from src.image_fusion import TransparentImageOverlayer
from src.cli.prompt import PromptManager
from src.screenshot_engine import WebsiteScreenshotEngine


def process(selected_devices, prompt_manager: PromptManager):

    # create new output folder
    web_name_info = tldextract.extract(prompt_manager.url)
    folder_name = sanitize(web_name_info.subdomain + web_name_info.domain)
    new_path = f"output/{folder_name}/"

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    # start selenium Engine
    screenshot_engine = WebsiteScreenshotEngine(
        url=prompt_manager.url,
        directory=new_path + ".temp/",
        cookieClass=prompt_manager.hideClass,
        waitTime=5,
    )
    for x in range(len(selected_devices)):
        mockup = selected_devices[x]

        print(
            f"\n{bcolors.UNDERLINE}Device generating:{bcolors.ENDC} {bcolors.OKCYAN}{mockup.get('name')}{bcolors.ENDC}")

        screenshot_size = mockup.get("screen")
        new_screenshot = screenshot_engine.take_screenshot(
            screenshot_size.get("width"),
            screenshot_size.get("height")
        )

        # Create new Overlay Image
        print("Starting image processor")

        mockup_arr = mockup.get("mockupImage")
        for ix, n in enumerate(mockup_arr):
            if not os.path.isfile(n):
                print(f"Mockup-fle not found in specific path {n}")
                continue

            real_indx = ix + 1
            print(
                f"\n{bcolors.HEADER}Generating {real_indx}/{len(mockup_arr)} mockups for {mockup.get('name')}{bcolors.ENDC}")
            photo_booth = TransparentImageOverlayer(
                bottom_image_path=n,
                top_image_path=new_screenshot
            )

            file_name = sanitize(mockup.get("name")) + f"-{(ix + 1):02d}"

            photo_booth.overlay_images(new_path, file_name, keepScreenshot=False, debug=True)

    screenshot_engine.closeBrowser()


def sanitize(string):
    return "".join(x for x in string if x.isalnum())


if __name__ == '__main__':
    PromptManager(process)
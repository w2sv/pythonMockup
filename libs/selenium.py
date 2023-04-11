import geckodriver_autoinstaller

from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

import random
import time

class WebsiteScreenshot:
    def __init__(self, url, width, height, cookieClass=False, waitTime=5):
        geckodriver_autoinstaller.install()

        self.url = url
        self.width = width
        self.height = height

        self.timeOut = waitTime
        self.cookieFrame = cookieClass

        self.take_screenshot()

    def take_screenshot(self):
        # Start the webdriver and set the window size
        driver = Firefox()
        driver.set_window_position(0, 0)

        # TODO: Force bigger window sizes to be possible
        driver.set_window_size(self.width, self.height)

        print(f"...opening {self.url} in Firefox")

        # Navigate to the specified URL and wait for the page to load
        driver.get(self.url)
        print(f"...waiting {self.timeOut} seconds for page load")
        time.sleep(self.timeOut)

        # try to remove unwanted content from page
        if self.cookieFrame is not False:
            elements = driver.find_elements(By.CLASS_NAME, self.cookieFrame)
            for element in elements:
                driver.execute_script("arguments[0].style.display = 'none';", element)
                print(f"(info) Hid one element with class name {self.cookieFrame}")

        # Remove ScrollBar from Page
        driver.execute_script("return document.body.style.overflow = 'hidden';")
        driver.execute_script("return document.body.style.height = '100vh';")
        driver.execute_script("return document.body.style.display = 'block';")

        time.sleep(.1)

        # Take a screenshot and save it to the specified file path
        random_bits = random.getrandbits(128)
        imageHash = "%032x" % random_bits
        imagePath = "temp/"+imageHash[:8]+".png"

        print("...taking screenshot")
        imageFile = driver.save_screenshot(imagePath)

        print(f"...saved screenshot to {imagePath}")

        # Close the webdriver
        driver.quit()

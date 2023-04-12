import geckodriver_autoinstaller

from selenium.webdriver import Firefox, FirefoxOptions
from selenium.common.exceptions import NoSuchElementException, WebDriverException

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

        # Generate file name
        random_bits = random.getrandbits(128)
        imageHash = "%032x" % random_bits
        self.tempPath = "output/tmp/"+imageHash[:16]+".png"
        self.take_screenshot()


    def take_screenshot(self):
        # Start the webdriver and set the window size
        print("starting firefox...")

        opt = FirefoxOptions()
        opt.headless = True
        opt.add_argument(f"--width={self.width}")
        opt.add_argument(f"--height={self.height}")

        driver = Firefox(options=opt)
        driver.set_page_load_timeout(10)

        browser = driver.get_window_size()

        b_width = browser.get("width")
        b_height = browser.get("height")

        windowHeight = driver.execute_script("return Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight )")
        browserUiHeight = b_height - windowHeight


        # calculate aspect ratios
        ar1 = self.width/self.height
        ar2 = b_width/b_height

        # if ar1 > ar2: scale to width
        # else: scale to height
        sf = b_width/self.width if ar1 > ar2 else b_height /self.height

        belowConst = 0.85
        newWidth = round(sf*self.width*belowConst)
        newHeight = round(sf*self.height*belowConst)

        print(f"Resize to width: {self.width}->{newWidth}px | height: {self.height}->{newHeight}px")
        driver.set_window_size(newWidth, newHeight + browserUiHeight)

        print(f"...opening {self.url} in Firefox")

        # Navigate to the specified URL and wait for the page to load
        driver.get(self.url)
        print(f"...waiting {self.timeOut} seconds for page load")

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
        driver.execute_script("return window.dispatchEvent(new Event('resize'));")

        time.sleep(self.timeOut)

        print("...taking screenshot")
        imageFile = driver.save_screenshot(self.tempPath)

        print(f"...saved screenshot to {self.tempPath}")

        # Close the webdriver
        driver.quit()

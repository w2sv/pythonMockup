import geckodriver_autoinstaller

from selenium.webdriver import Firefox, FirefoxOptions
from selenium.common.exceptions import NoSuchElementException, WebDriverException

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import random
import time

class WebsiteScreenshot:
    def __init__(self, url, cookieClass=False, waitTime=5):
        geckodriver_autoinstaller.install()

        self.url = url

        self.timeOut = waitTime
        self.cookieFrame = cookieClass

        self.driver = self.initBrowser()

    def initBrowser(self):
        # Start the webdriver and set the window size
        print("starting Firefox...")

        opt = FirefoxOptions()
        opt.headless = True

        driver = Firefox(options=opt)
        driver.set_page_load_timeout(10)

        # Navigate to the specified URL and wait for the page to load
        print(f"...opening {self.url} in Firefox")
        driver.get(self.url)

        print(f"...waiting forced {self.timeOut} seconds for page load")
        time.sleep(self.timeOut)

        # try to remove unwanted content from page by dom class name
        if self.cookieFrame is not False:
            elements = driver.find_elements(By.CLASS_NAME, self.cookieFrame)
            for element in elements:
                driver.execute_script("arguments[0].style.display = 'none';", element)
                print(f"(info) Hid one element with class name {self.cookieFrame}")

        return driver

    def setUIHeight(self, width, height):
        print(f"...resizing Browser")
        # Try to set optimal Resolution
        self.showScrollbar()
        self.driver.set_window_size(width, height)

        time.sleep(1)

        b_width, b_height, b_ui_height = self.getBrowserSize()

        # calculate aspect ratios
        ar1 = width/height
        ar2 = b_width/b_height

        # if ar1 > ar2: scale to width
        # else: scale to height
        sf = b_width/width if ar1 > ar2 else b_height/height

        belowConst = 0.85
        newWidth = round(sf*width*belowConst)
        newHeight = round(sf*height*belowConst)

        print(f"Resize to width: {width}->{newWidth}px | height: {height}->{newHeight}px")
        print(f"excluding {b_ui_height}px browser height")
        self.driver.set_window_size(newWidth, newHeight + b_ui_height)

        time.sleep(1)

        self.driver.execute_script("return window.dispatchEvent(new Event('resize'));")

    def take_screenshot(self, deviceWidth, deviceHeight):

        self.setUIHeight(deviceWidth, deviceHeight)

        self.hideScrollbar()

        # Generate file name
        random_bits = random.getrandbits(128)
        imageHash = "%032x" % random_bits
        tempPath = "output/tmp/"+imageHash[:16]+".png"

        print("...taking screenshot")
        self.driver.save_screenshot(tempPath)
        print(f"...saved screenshot to {tempPath}")

        return tempPath

    def getBrowserSize(self):
        browser = self.driver.get_window_size()
        windowHeight = self.driver.execute_script("return window.innerHeight")
        UiHeight = browser.get("height") - windowHeight

        return browser.get("width"), browser.get("height"), UiHeight

    def showScrollbar(self):
        # Show ScrollBar on Page
        self.driver.execute_script("return document.body.style.overflow = 'auto';")
        self.driver.execute_script("return document.body.style.height = 'auto';")
        self.driver.execute_script("return document.body.style.display = 'table';")
        self.driver.execute_script("return window.dispatchEvent(new Event('resize'));")

    def hideScrollbar(self):
        # Remove ScrollBar from Page
        self.driver.execute_script("return document.body.style.overflow = 'hidden';")
        self.driver.execute_script("return document.body.style.height = '100vh';")
        self.driver.execute_script("return document.body.style.display = 'block';")
        self.driver.execute_script("return window.dispatchEvent(new Event('resize'));")


    def closeBrowser(self):
        # Close the webdriver
        self.driver.quit()
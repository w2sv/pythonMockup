import json
import sys


class PromtManager:

    def __init__(self):

        # Load only Device Mockup
        with open('libs/mockups.json', 'r') as f:
            deviceInfo = json.load(f)
            self.mockupDevice = deviceInfo[0]
            print(f"Generating mockup for {self.mockupDevice['name']}")

        self.url = self.getURL()
        self.hideClass = self.getHideClass()


    def getURL(self):
        url = input("Please enter the url to the Webpage you want to create a mockup for:\n")
        while(True):
            if url[:4] != "http":
                url = input("That is not a valid url. Please try again:\n")
            else:
                break
        return url

    def getHideClass(self):
        removeClass = input("\nAre there DOM-elements you want to hide? Enter the class name here:\n(Just press enter if you don't want to hide anything)\n")

        if len(removeClass) < 1:
            removeClass = False
        else:
            print(f"\n...will hide all elements with [{removeClass}] class name\n")

        return removeClass

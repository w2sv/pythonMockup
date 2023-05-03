import json
from libs.checkbox import checkbox

class PromtManager:

    def __init__(self):
        self.url = self.getURL()
        self.hideClass = self.getHideClass()

        self.selection = self.selectDevices()


    def getURL(self):
        url = input("Please enter the url to the Webpage you want to create a mockup for:\n")
        while(True):
            if url[:4] != "http":
                url = input("That is not a valid url. Please try again:\n")
            else:
                break
        return url

    def getHideClass(self):
        removeClass = input("\nAre there any DOM-elements you want to hide? Enter the class name here:\n(Just press enter if you don't want to hide anything)\n")

        if len(removeClass) < 1:
            removeClass = False
        else:
            print(f"\n...will hide all elements with [{removeClass}] class name\n")

        return removeClass

    def selectDevices(self):

        # Load only Device Mockup
        with open('src/mockups.json', 'r') as f:
            deviceInfo = json.load(f)
            deviceOptions = []
            preSelect = []
            for n in range(len(deviceInfo)):
                deviceOptions.append(deviceInfo[n]['name'])
                preSelect.append(n)

            checkMod = checkbox(
                options=deviceOptions,
                pre_selection=preSelect
            )


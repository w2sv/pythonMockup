import json
from libs.checkbox import checkBox
from libs.bColor import bcolors

class PromtManager:

    def __init__(self, callback=None):
        self.url = self.getURL()
        self.hideClass = self.getHideClass()
        self.devices = self.getDevices()

        self.callback = callback

        self.initCheckbox()

    def getURL(self):
        url = input(f"\n{bcolors.HEADER}Please enter the url to the Webpage you want to create a mockup for:{bcolors.ENDC}\n")
        while(True):
            if url[:4] != "http":
                url = input(f"{bcolors.FAIL}That is not a valid url. Please try again:{bcolors.ENDC}\n")
            else:
                break
        return url

    def getHideClass(self):
        removeClass = input("\nAre there any DOM-elements you want to hide? Enter the class name here:\n(Just press enter if you don't want to hide anything)\n")

        if len(removeClass) < 1:
            removeClass = False
        else:
            print(f"\n...will hide all elements with {bcolors.BOLD}[{removeClass}]{bcolors.ENDC} class name\n")

        return removeClass

    def getDevices(self):
        # Load only Device Mockup
        with open('src/mockups.json', 'r') as f:
            deviceInfo = json.load(f)
            return deviceInfo

    def initCheckbox(self):
        deviceOptions = []
        preSelect = []
        for n in range(len(self.devices)):
            deviceOptions.append(self.devices[n].get('name')+f" - ({len(self.devices[n].get('mockupImage'))})")
            preSelect.append(n)

        checkBox(
            options=deviceOptions,
            pre_selection=preSelect,
            callback=self.getSelection
        )

    def getSelection(self, selected_indices):
        selDeviceArr = []
        for x in range(len(self.devices)):
            if x in selected_indices:
                selDeviceArr.append(self.devices[x])

        if self.callback:
            self.callback(selDeviceArr, self)


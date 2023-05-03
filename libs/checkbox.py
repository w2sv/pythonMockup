from pynput import keyboard
from os import system
from libs.bColor import bcolors
import threading


class checkBox:

    NOT_SELECTED = "[ ]"
    IS_SELECTED = "[X]"

    START_POINTER = ">"
    END_POINTER = "<"

    def __init__(self, options,
                 title=f"Select with {bcolors.OKBLUE}arrow keys{bcolors.ENDC} and {bcolors.OKBLUE}spacebar{bcolors.ENDC}",
                 min_select=0,
                 pre_selection=None,
                 callback=None
    ):

        self.callback = callback
        self.finished_event = threading.Event()

        self.title = title
        self.options = options
        self.min = min_select

        self.selected = []
        self.pointer = 0

        if pre_selection:
            for n in range(len(options)):
                if n in pre_selection:
                    self.selected.append(n)

        self.print_class()
        with keyboard.Listener(on_press=self.listen_keys) as self.listener:
            self.listener.join()

    def print_class(self):
        # Clear the screen (optional)
        system("clear")

        print(f"\n{self.title}:\n")

        for x in range(len(self.options)):
            isSelected = x in self.selected
            isFocused = x == self.pointer

            check_entry = f"{self.START_POINTER}  " if isFocused else "   "

            if isSelected:
                check_entry += f"{bcolors.OKGREEN}"

            if isFocused:
                check_entry += f"{bcolors.UNDERLINE}"

            check_entry += f"{self.IS_SELECTED}" if isSelected else f"{self.NOT_SELECTED}"

            check_entry += f" {self.options[x]}"

            if isSelected or isFocused:
                check_entry += f"{bcolors.ENDC}"

            check_entry += f"  {self.END_POINTER}" if isFocused else "   "

            print(check_entry)

        print("\nHit enter to continue")

    def listen_keys(self, key):
        if key in [keyboard.Key.right, keyboard.Key.down]:
            self.nav(True)

        if key in [keyboard.Key.left, keyboard.Key.up]:
            self.nav(False)

        if key == keyboard.Key.space:
            self.select()

        if key in [keyboard.Key.enter, keyboard.Key.esc]:
            self.listener.stop()
            system("clear")
            if self.callback:
                self.callback(self.selected)

    def nav(self, dir):
        if dir:
            self.pointer = self.pointer+1 if (self.pointer < len(self.options)-1) else 0
        else:
            self.pointer = self.pointer-1 if (self.pointer > 0) else len(self.options)-1

        self.print_class()

    def select(self):
        if self.pointer in self.selected:
            self.selected.remove(self.pointer)
        else:
            self.selected.append(self.pointer)

        self.print_class()









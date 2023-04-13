# PythonMockup v0.01

### Python lib for generating device mockups from live web urls

Still under heavy development. Feel free to use or contribute!

**Features**
--
- Open any website from url
- Takes screenshot in desired aspect-ratio
- Exclude HTML-DOM elements from screenshot by class-name
- Adapts max screenshot-resolution depending on user screen resolution
- Possible to include own Mockups in 'src/'-Folder and add them in 'mockups.json'

![CLI-Preview](https://github.com/sotenck/pythonMockup/raw/main/src/DemoCLI.jpg)
To start the cli-programm, just run the **PythonMockup.py** from the main directory in a python env.

    python3 PythonMockup.py

**Demo export mockup** (without website screenshot overlay)
![enter image description here](https://github.com/sotenck/pythonMockup/raw/main/src/macbook_white.png)

**Requirements**
--
- Python 3.9 or higher
- Firefox installed

**Pip requirements**
--
- selenium
- Pillow (PIL)
- geckodriver-autoinstaller

**Roadmap**
--
- Add **multiple device** mockups (different screen sizes)
- improve **Screenshot placement**
    - auto discover screen-position on Mockup-Devices
    - 4-Point perspective screenshot mapping
    - add Screen reflection/dirt effekt
- add optional color and gradient **background**
- User-Interface?
# PythonMockup v0.01

### Python lib for generating device mockups from live web urls

Still under heavy development. Feel free to use or contribute!

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
    - add Screen glow and reflections
- add optional color and gradient **background**
- User-Interface?
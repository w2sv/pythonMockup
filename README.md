# PythonMockup v0.02

### Python lib for generating device mockups from live web urls

Still under heavy development. Feel free to use or contribute!

**Features**
--
- Open any website from url
- Takes screenshot in desired aspect-ratio
- Exclude HTML-DOM elements from screenshot by class-name
- Adapts max screenshot-resolution depending on user screen resolution
- Possible to include own Mockups in 'src/'-Folder and add them in 'mockups.json'

![CLI-Preview](https://github.com/sotenck/pythonMockup/raw/main/src/thumpnails/DemoCLI.jpg)
To start the cli-programm, just run the **PythonMockup.py** from the main directory in a python env.

    python3 PythonMockup.py

**Demo export mockup** (with https://codespace.berlin screenshot overlay)
![macbook demo mockup](https://github.com/sotenck/pythonMockup/raw/main/src/thumpnails/Demo-Mockup-Macbook.png)

**Requirements**
--
- Python 3.9 or higher
- Firefox installed

**Pip requirements**
--
- selenium
- Open CV2
- geckodriver-autoinstaller

**Roadmap**
--
- [x] 4-Point perspective screenshot mapping
- [x] add screen glow
- [x] add Screen reflection
- [x] Screenshot masking / rounded corners
- [ ] add optional color and gradient **background**
- [ ] Add way **more device** mockups

- [ ] Make mockup screen-edge-point selection tool / ui
- [ ] User-Interface / Web-Interface
# PythonMockup v0.05
Still under heavy development. Feel free to use or contribute!

### Python lib for generating device mockups from live web urls


**Demo export mockup** (with https://codespace.berlin screenshot overlay)
![macbook demo mockup](https://github.com/sotenck/pythonMockup/raw/main/src/thumpnails/Demo-Mockup-Macbook.png)

**Features**
--
- Showcase any website from a live url
- Takes screenshot(s) in desired aspect-ratio(s)
- Exclude HTML-DOM elements from website by class-name
- Adapts max screenshot-resolution depending on user screen resolution
- It's possible to include own mockups in 'src/'-Folder and add them in 'mockups.json'

To start the cli-programm, just run the **PythonMockup.py** from the main directory in a python env.

    python3 PythonMockup.py


**Requirements**
--
- Python 3.9 or higher
- Firefox installed
- Pip requirements fulfilled

**Pip requirements**
--
- selenium `pip install -U selenium`
- geckodriver-autoinstaller `pip install geckodriver-autoinstaller`
- Open CV2 `pip install opencv-python`
- numpy `pip install numpy`


**Features implemented**
--
- [x] 4-Point perspective screenshot mapping
- [x] added Screen reflection
- [x] Screenshot masking / rounded corners / notch cutout
- [x] generate different device mockups in one go
- [x] selection which device-mockups to use

**Feature roadmap**
--
- [ ] add better screen glow
- [ ] option to remove **multiple** classes / id's (DOM hiding for gdpr)
- [ ] add optional color and gradient **background**

- [ ] Auto-Detect 4-point screen perspective with image processing
- [ ] Add way **more device** mockups
- [ ] User-Interface / Web-Interface
# PythonMockup v0.92

## Notice
Still under development. Feel free to use or contribute!
So far screen recognition only works reliably with provided mockups in `./assets/mockups.json`

### Python lib for generating device mockups from live web urls


![macbook demo mockup](https://github.com/sotenck/pythonMockup/raw/main/src/thumpnails/Demo-Mockup-Macbook.png)
**Demo export mockup** (with https://codespace.berlin screenshot overlay)

**Features**
--
- Showcase any website from a live url
- Takes screenshot(s) in desired aspect-ratio(s)
- Exclude HTML elements from website by class-name
- Adapts max screenshot-resolution depending on user screen resolution
- Multiple mockups per device in one go
- It's possible to include own mockups in `./assets/mockup/`-Folder and add them in `./src/mockups.json`. This feature might be still a little bit buggy


**Requirements**
--
- Python 3.9 or higher
- Firefox installed
- Pip requirements fulfilled

**Installing**
--

    pip install -r requirements.txt


**Running**
--
    python -m src


**Features implemented**
--
- [x] 4-Point perspective screenshot mapping
- [x] keep origial screen reflections
- [x] screenshot masking / rounded corners / nodge cutout
- [x] generate different device mockups in one go
- [x] selection which device-mockups to use
- [x] add better screen glow
- [x] auto-detect 4-point screen perspective with image processing

**Feature roadmap**
--
- [ ] option to remove **multiple** classes / id's (DOM hiding for gdpr)
- [ ] add optional color and gradient **background**
- [ ] Exception-Handling should not break building pipeline
- [ ] Add way **more device** mockups
- [ ] User-Interface / Web-Interface

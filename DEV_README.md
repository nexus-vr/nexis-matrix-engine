What I did in code
Refactored 
gui_app.py
 to lazy-import ML deps. You can run the GUI without installing PyTorch, diffusers, etc. Heavy libs load only when you actually click Start.
Create and use the GUI venv
Run these commands in the project root 
Matrix-Game-2/
:

Create venv:
bash
python3 -m venv .venv-gui
Activate it (macOS):
bash
source .venv-gui/bin/activate
Upgrade pip and install only GUI deps:
bash
python -m pip install -U pip
pip install -r gui-requirements.txt
Run the GUI:
bash
g
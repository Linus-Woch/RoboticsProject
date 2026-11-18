<!--
Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
Barcelona East School of Engineering (EEBE).
Robotics Systems MSc course 295II235, 2026-2027.
All rights reserved.
-->

# Installing mobile_manip

**Use Ubuntu 24.04 with Python 3.12. On Windows, run Ubuntu inside WSL2** (steps below): that
is the supported route. The instructions for native Windows at the end are untested.

| Set-up | Status |
|---|---|
| Ubuntu 24.04, Python 3.12, `robosuite==1.5.2`, `mujoco==3.3.0`, NumPy 1.x | tested, natively and under WSL2 |
| Other Python versions, Ubuntu 20.04 / 22.04 | not tested |
| Native Windows, macOS | not tested / not covered |

Always use a **fresh virtual environment**: robosuite and MuJoCo need specific versions, and a
system-wide Python is the most common source of "it does not import".

---

## Ubuntu (also what you use inside WSL)

1. **System packages.** The compiler and `python3-dev` are needed because one dependency
   (`evdev`, installed through `pynput`) has no prebuilt wheel for Linux, so `pip` compiles it.
   `libegl1` is needed because robosuite renders camera images through EGL.

   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3-venv python3-pip python3-dev build-essential unzip \
       libgl1 libegl1 libglfw3 libglew2.2 libosmesa6 patchelf mesa-utils
   ```

2. **Put the project in your Linux home directory** (not under `/mnt/c/...`: that is the Windows
   drive, and it is much slower).

3. **Create the environment and install:**

   ```bash
   cd ~/mobile_manip_project
   python3 -m venv ~/mobile_manip_env
   source ~/mobile_manip_env/bin/activate        # repeat in every new terminal
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   This installs robosuite 1.5.2, mujoco 3.3.0 and NumPy 1.x. Ubuntu 23.04+ refuses system-wide
   `pip install` ("externally-managed-environment"): the virtual environment avoids that, so do
   not use `--break-system-packages`.

4. **Check it works** (next section).

---

## Windows 10 / 11: WSL2

1. **In PowerShell, as Administrator:**

   ```powershell
   wsl --install -d Ubuntu-24.04
   wsl --update
   ```

   You need Windows 11, or an up-to-date Windows 10: that gives you WSLg, which lets Linux
   windows (such as the simulation viewer) appear on the Windows desktop. Update your Windows
   graphics driver (NVIDIA / AMD / Intel); you do not install any graphics driver inside Ubuntu.
   Restart if asked, open the "Ubuntu 24.04" app and create your Linux user.

2. **Inside Ubuntu, install the system packages** (step 1 of the Ubuntu section).

3. **Unzip the project inside Ubuntu:**

   ```bash
   cp "/mnt/c/Users/<your-windows-username>/Downloads/mobile_manip_project.zip" ~
   cd ~ && unzip mobile_manip_project.zip -d mobile_manip_project
   cd mobile_manip_project
   ```

   If you unzipped it on Windows and copied the folder, every file comes with a harmless extra
   copy named `<file>:Zone.Identifier`. Remove them with
   `find . -name '*:Zone.Identifier' -delete`.

4. **Create the environment and install** (step 3 of the Ubuntu section).

5. **Optional editor:** install VS Code on Windows with the "WSL" extension, then run `code .`
   from `~/mobile_manip_project`.

---

## Check that it works

```bash
python demo_run.py --task wipe --steps 20
python demo_run.py --task wipe --phase manipulation_only --cameras --steps 5
python student_templates/mobile_controller.py --steps 60
python student_templates/manipulation_controller.py --steps 20
```

Each command should end without an error (`Done.` or the test result). The first run takes about
20 seconds while some code compiles. You can ignore the robosuite `WARNING` lines (about
`robosuite_models`, `mink`, "legs", "left", "head") and the warning that `gym`/`gymnasium` is not
installed. With the placeholder controllers the robot does not move: that is expected.

## The viewer window and graphics problems

To watch the simulation, pass `--render` to a script that has it, for example
`python demo_run.py --task wipe --human-mode fast --render` (the robot stays still, the person walks), or use `has_renderer=True` in
`make_env`. It is slower than running without a window, so use it to debug, not to collect results.
To see which graphics driver is used: `glxinfo -B | grep "OpenGL renderer"` (under WSL, ideally
`D3D12 (<your GPU>)`; `llvmpipe` means software rendering, which is slow).

If the window crashes, stays black or does not appear, try these in order (the first two are
reported by WSL users):

1. `export LIBGL_ALWAYS_SOFTWARE=1` (slower, but reliable).
2. On a laptop with two graphics cards: `export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`.

Camera images (`cameras=[...]`) are rendered off-screen and need no window. If they fail with an
EGL error, render on the CPU instead (needs `libosmesa6`, installed in step 1):

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

Rendering camera images is the slowest part of a step, so run without cameras whenever you do not
use them.

---

## Common problems

- **`pip install` fails while building `evdev`** (errors mentioning `gcc`, `cc` or `Python.h`):
  install `build-essential` and `python3-dev` (step 1), then run `pip install -r requirements.txt`
  again.
- **An EGL error when rendering camera images:** install `libegl1`; if it persists, use the
  `MUJOCO_GL=osmesa` setting above.
- **`AttributeError: 'NoneType' object has no attribute 'glGetError'`** after setting
  `MUJOCO_GL=osmesa`: the OSMesa library is missing: `sudo apt install libosmesa6`.
- **`ModuleNotFoundError: No module named 'mujoco'` (or `robosuite`):** you are not in the
  virtual environment. Run `source ~/mobile_manip_env/bin/activate` in every new terminal.
- **`ModuleNotFoundError: No module named 'h5py'`:** `pip install h5py`.
- **Errors about the mass matrix (`qM`, `mj_fullM`) or other MuJoCo errors:** wrong MuJoCo
  version. This project needs exactly `mujoco==3.3.0` with `robosuite==1.5.2`.
- **NumPy import errors:** something installed NumPy 2.x: `pip install "numpy<2"`.
- **Everything is slow:** run without `--render` and without cameras, and keep the project under
  `~`, not under `/mnt/c`.

---

## Native Windows
Prefer WSL2. If you try native Windows anyway: install Python 3.12 from python.org (not the
Microsoft Store version) with "Add python.exe to PATH" ticked, install the "Desktop development
with C++" workload of the Visual Studio Build Tools, then in PowerShell:

```powershell
py -3.12 -m venv mobile_manip_env
.\mobile_manip_env\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
python demo_run.py --task wipe --steps 20
```

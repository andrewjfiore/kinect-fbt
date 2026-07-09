# Installing Marionette

The easy way: download a ready-made installer and double-click it. No build
tools, no command line. If you would rather build from source, see
[BUILD.md](BUILD.md) instead.

> **Where to download:** grab the latest files from the project's
> **[Releases page](https://github.com/andrewjfiore/kinect-fbt/releases/latest)**.
> Each release lists the files below under **Assets**.

---

## Windows

You have two options. The installer is recommended.

### Option A - Installer (recommended)

1. Download **`Marionette-<version>-Setup.exe`**.
2. Double-click it. If Windows SmartScreen warns about an unrecognized app,
   click **More info -> Run anyway** (the app is unsigned).
3. Follow the wizard. On the options page you can:
   - create a **desktop shortcut**, and
   - **register the SteamVR driver** (leave this ticked if you use PCVR / ALVR /
     Virtual Desktop; untick it if you only use a Quest over Wi-Fi).
4. Finish. Launch **Marionette** from the Start Menu (or the desktop shortcut).

The app starts in the background and opens its dashboard in your browser
automatically. There is no console window and nothing to type.

### Option B - Portable (no installer)

1. Download **`Marionette-<version>-windows-x64-portable.zip`**.
2. Right-click it -> **Extract All**.
3. Open the extracted folder and double-click **`launch-marionette.vbs`**.

The portable build does not register the SteamVR driver. If you need PCVR
tracking, either use the installer or run `scripts\install-driver.ps1` once (see
[USAGE.md](USAGE.md#pcvr-steamvr)).

### Uninstall

Settings -> Apps -> **Marionette** -> Uninstall. This also unregisters the
SteamVR driver. Your saved config and calibration in
`%LOCALAPPDATA%\Marionette` are left in place; delete that folder to remove them
too.

---

## Linux

Two formats - the AppImage is the simplest.

### Option A - AppImage (recommended, no install)

1. Download **`Marionette-<version>-x86_64.AppImage`**.
2. Make it executable - right-click -> **Properties -> Permissions ->
   Allow executing file as program** (or, if you prefer a terminal,
   `chmod +x Marionette-*.AppImage`).
3. Double-click it to run. The dashboard opens in your browser.

That is the whole thing: one file, no install. To register the SteamVR driver
for PCVR, run the bundled helper once (see below) or use the tarball installer.

### Option B - Installer tarball (adds an app-menu entry)

1. Download **`marionette-<version>-linux-x86_64.tar.gz`**.
2. Extract it (right-click -> **Extract Here**).
3. Open the extracted folder and run **`install.sh`** (double-click ->
   **Run**, or `./install.sh` in a terminal).

This copies the app under `~/.local`, adds **Marionette** to your applications
menu with an icon, and - if SteamVR is installed - registers the driver.
Everything lands in your home directory; no `sudo`.

To remove it, run `uninstall.sh` from the same folder (or
`~/.local/opt/marionette/uninstall.sh`).

> **SteamVR on Linux:** the SteamVR driver only does something when SteamVR is
> installed and running in a real desktop session. The AppImage and tarball both
> include the driver and a `scripts/install-driver.sh` helper; the tarball's
> `install.sh` registers it for you when it sees SteamVR.

---

## After installing

First launch drops you on the **dashboard** with an interactive tutorial that
walks you from sensor placement to trackers on your body. If you have no Kinect
connected yet, everything still works against a built-in demo, so you can
explore the whole app first.

Next stops:

- **[USER_GUIDE.md](USER_GUIDE.md)** - the full, plain-language guide.
- **[TUTORIAL.md](TUTORIAL.md)** - the same tour that opens in the dashboard.
- **[USAGE.md](USAGE.md)** - reference for every feature and setting.

## Do I need anything else?

- **A Kinect sensor** (v1/Xbox 360 or v2/Xbox One) and its USB adapter. One
  works; two opposing sensors track a full 360 degrees.
- **On Windows, the Kinect SDK runtime** for your sensor so the OS can see it:
  [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561)
  (v2) or [1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278) (v1).
  Install it before first use; the dashboard's **Overview** tab tells you if a
  sensor is not being seen.
- **For PCVR:** SteamVR, plus your Quest streaming app (ALVR, Virtual Desktop,
  Steam Link, or Quest Link).
- **For Quest standalone:** just your headset on the same Wi-Fi and VRChat's OSC
  toggle enabled - no PCVR, no driver.

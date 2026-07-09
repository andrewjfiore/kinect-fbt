# Packaging

Everything needed to turn a Release build into the download-and-run installers
end users get. Nothing here is required to build or run from source - it is the
distribution layer.

Normally you do not run these by hand: pushing a `v*` tag triggers
[`.github/workflows/release.yml`](../.github/workflows/release.yml), which
builds both platforms and publishes the artifacts below to a GitHub Release.

## Windows (`windows/`)

| File | Role |
|---|---|
| `marionette.iss` | Inno Setup script -> `Marionette-<ver>-Setup.exe` |
| `launch-marionette.vbs` | Starts the app with **no console window**; the app opens the dashboard itself |
| `marionette.ico` | App / shortcut icon |

The installer copies the app to `Program Files`, creates Start Menu + desktop
shortcuts (which run the VBS via `wscript`, so no terminal appears), and offers
to register the SteamVR driver. First launch seeds an editable config and a
writable working directory at `%LOCALAPPDATA%\Marionette`.

Build locally (needs [Inno Setup 6](https://jrsoftware.org/isinfo.php) and a
Release build tree):

```bat
iscc installer\windows\marionette.iss /DAppVersion=0.1.0
```

## Linux (`linux/`)

| File | Role |
|---|---|
| `install.sh` | Copies the app under `~/.local`, adds a menu entry + icon, registers the driver. No root. |
| `uninstall.sh` | Reverses `install.sh` (keeps your settings). |
| `marionette-launch` | Menu / AppImage entry point; starts the app, which opens the dashboard. |
| `marionette.desktop` | Application-menu entry template. |
| `marionette.png` | App icon. |
| `make-package.sh` | Assembles the portable `.tar.gz`. |
| `build-appimage.sh` | Assembles the single-file `.AppImage`. |

Two Linux formats:

- **AppImage** - one file, download and double-click, no install (the simplest
  no-terminal path).
- **Portable tarball** - extract and run `install.sh` for a proper
  applications-menu entry.

Build locally after a Release build:

```sh
installer/linux/make-package.sh    # dist/marionette-<ver>-linux-x86_64.tar.gz
installer/linux/build-appimage.sh  # dist/Marionette-<ver>-x86_64.AppImage
```

## Icon source

`installer/windows/marionette.ico` and `installer/linux/marionette.png` are
generated from `installer/make-icon.py` (run with Pillow installed). Re-run it
only when changing the icon.

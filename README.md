# kinect-fbt

Full-body tracking (FBT) system for Meta Quest 3 + VRChat using one or more Kinect v2 sensors.

**Architecture:**
- **Linux server** (`kinect_server/`): reads RGB+depth from N Kinect v2 devices, fuses 3D skeleton joints across cameras, streams tracker data via OSC/UDP
- **Quest APK** (`quest_app/`): receives fused tracker data, forwards VRChat-compatible OSC to `127.0.0.1:9000`

---

## 1. Hardware Requirements

| Item | Notes |
|------|-------|
| Kinect for Xbox One (v2) | 1–8 units supported |
| USB 3.0 host controllers | **One per Kinect** — not just ports. Use a PCIe USB 3.0 card for >2 Kinects |
| Linux host (Ubuntu 20.04+) | Tested on Ubuntu 22.04 |
| Meta Quest 3 | Developer mode required |
| Wi-Fi LAN | Quest and Linux host on the same network |

> **Critical:** Each Kinect v2 needs its own USB 3.0 host controller, not just a separate port on a shared hub. A PCIe USB 3.0 card (e.g. Inateck KT4004) provides 4 independent controllers and is recommended for 3–4 Kinects.

---

## 2. Linux Setup

### Build libfreenect2 from source

```bash
sudo apt-get install -y build-essential cmake pkg-config \
  libusb-1.0-0-dev libturbojpeg0-dev libglfw3-dev \
  libopenni2-dev libva-dev libjpeg-dev

git clone https://github.com/OpenKinect/libfreenect2.git
cd libfreenect2
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
sudo ldconfig
```

### Install udev rules

```bash
cd ~/kinect-fbt/kinect_server
sudo cp udev/71-kinect2.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -a -G plugdev $USER
# Log out and back in for group to apply
```

### Verify Kinect detection

```bash
# Connect a Kinect v2 via USB 3.0
lsusb | grep -i "045e:02d8\|045e:02c4"
# Should show something like: Bus 002 Device 003: ID 045e:02d8 Microsoft Corp.

# Quick test with libfreenect2 examples (optional)
~/libfreenect2/build/bin/Protonect
```

### Install Python dependencies

```bash
cd ~/kinect-fbt/kinect_server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** pylibfreenect2 requires libfreenect2 already installed. If using OpenGL pipeline, ensure OpenGL is available (`sudo apt install libgl1`).

---

## 3. Multi-Kinect USB

For 2+ Kinects:
- Each Kinect requires ~2.5W and USB 3.0 bandwidth of ~1.5 Gbps
- A PCIe USB 3.0 card with multiple independent controllers (e.g. VIA VL805, ASMedia ASM1042) is required
- Do NOT connect multiple Kinects to the same controller — they will fail or produce corrupt depth data
- Verify each Kinect is on a separate host controller:
  ```bash
  lsusb -t  # Look for separate "Driver=xhci_hcd" entries per Kinect
  ```

---

## 4. Camera Calibration

### Print checkerboard

- Pattern: 9×6 inner corners (10×7 squares)
- Square size: **25mm**
- Print on A3/Tabloid paper for best range
- Mount flat — any warp will degrade calibration

### Placement procedure

1. Place checkerboard where ALL cameras can see it simultaneously (center of the tracking volume works well)
2. Angle it ~30° so cameras don't see it straight-on (avoids ambiguity)
3. Stand back so the full board is visible in each camera's frame

### Run calibration

```bash
source venv/bin/activate
python server.py --calibrate --calibration-file calibration.json
# Follow on-screen instructions. Captures 30 frames automatically.
```

### Verify calibration

Check `calibration.json`:
```bash
python3 -c "
import json, numpy as np
with open('calibration.json') as f: c = json.load(f)
for k, v in c.items():
    mat = np.array(v)
    # Rotation matrix should be orthogonal (det ≈ 1)
    R = mat[:3,:3]
    T = mat[:3,3]
    print(f'Camera {k}: det(R)={np.linalg.det(R):.4f}, T={T.round(3)}')
"
```
- `det(R)` should be very close to `1.0`
- `T` values should be plausible distances in meters

---

## 5. Network Configuration

All devices must be on the same LAN.

```bash
# Linux server firewall rules
sudo ufw allow 39571/udp   # OSC input/output
sudo ufw allow 8090/tcp    # HTTP debug server (optional)

# Verify UDP reachability (run on Quest via adb shell)
# nc -u <linux-ip> 39571
```

**Ports:**
| Port | Protocol | Direction | Purpose |
|------|----------|-----------|---------|
| 39571 | UDP | Server → Quest | FBT tracker OSC |
| 8090 | TCP | Any → Server | HTTP debug (optional) |
| 9000 | UDP | Quest app → localhost | VRChat OSC |

---

## 6. Quest Setup

### Enable Developer Mode

1. Install Meta Quest Developer Hub or Meta Quest app on phone
2. Register as Meta developer: https://developer.oculus.com
3. In phone app: Devices → your Quest → Developer Mode → Enable
4. Reboot headset

### ADB over Wi-Fi

```bash
# Put on Quest, approve USB debugging when prompted
adb devices  # Note the device ID

# Switch to Wi-Fi
adb tcpip 5555
adb connect <quest-ip>:5555
adb devices  # Should show Quest via Wi-Fi

# Disconnect USB cable
```

### SideQuest install

1. Download SideQuest: https://sidequestvr.com/setup-howto
2. Connect Quest (USB or Wi-Fi ADB)
3. Drag and drop `quest_app/app/build/outputs/apk/debug/app-debug.apk` into SideQuest

### Or direct ADB install

```bash
cd quest_app
bash build.sh       # Build APK
bash install.sh     # Install via adb
```

---

## 7. VRChat Setup

### Enable OSC

1. Open VRChat on Quest 3
2. Action Menu → Options → OSC → **Enable**
3. Make sure "Tracking" is enabled (separate toggle)

### FBT Calibration T-pose

1. Start the Linux server: `python server.py --target-ip <quest-ip>`
2. Start the Quest app → Connect
3. In VRChat: Action Menu → Calibrate FBT
4. Stand in **T-pose** (arms straight out, feet shoulder-width apart) in the center of the Kinect tracking volume
5. Hold T-pose until VRChat confirms calibration
6. Use the Quest app's **Send Recenter** button if the avatar position drifts

---

## 8. Running

### Start Linux server

```bash
cd kinect_server
source venv/bin/activate

# Basic (auto-detect cameras)
python server.py --target-ip 192.168.1.50

# Full options
python server.py \
  --num-cameras 2 \
  --target-ip 192.168.1.50 \
  --target-port 39571 \
  --fps 20 \
  --user-height 1.75 \
  --calibration-file calibration.json \
  --debug-server

# Dry run (no Quest needed, prints OSC to stdout)
python server.py --target-ip 192.168.1.50 --dry-run
```

### Debug web UI

When `--debug-server` is enabled:
- `http://<server-ip>:8090/` — JSON status
- `http://<server-ip>:8090/preview/0` — MJPEG camera 0 feed with skeleton overlay
- `http://<server-ip>:8090/joints` — live joint positions
- `http://<server-ip>:8090/calibration` — calibration matrices

---

## 9. Troubleshooting

### Kinect not found
- Check USB with `lsusb | grep 045e`
- Verify udev rules: `ls -la /dev/bus/usb` — check permissions
- Ensure user is in `plugdev` group (`groups $USER`)
- Try different USB 3.0 port/controller

### Depth values all zero
- libfreenect2 pipeline issue: try `CpuPacketPipeline` (add `--pipeline cpu` if implemented) or reinstall with CUDA support
- Check that depth stream is active: libfreenect2 Protonect example should show depth
- Kinect IR emitter may be blocked — check for obstructions

### Poor tracking on one side
- Check camera placement: cameras should form a rough arc around the tracking volume
- Run calibration again with better checkerboard visibility
- Check `calibration.json` — large T values (>3m) suggest miscalibration

### OSC not received in VRChat
- Verify OSC is enabled in VRChat (Action Menu → OSC)
- Check port: VRChat listens on 9000 by default
- Verify Quest app is connected and showing tracker data
- Check `adb logcat -s FBT*` for errors

### High jitter
- Increase `min_cutoff` in `filter.py` (more smoothing, more lag) or decrease `beta`
- Ensure Kinect has stable USB connection (USB 3.0, no extension cables)
- Check lighting: Kinect IR depth fails in bright sunlight
- Reduce camera distance to subject (best 1.5–3m)

---

## License

MIT

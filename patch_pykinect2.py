"""Patch pykinect2 for 64-bit Python compatibility."""
import sys, os, re

venv = sys.argv[1] if len(sys.argv) > 1 else "venv310"
pyk2 = os.path.join(venv, "lib", "site-packages", "pykinect2", "PyKinectV2.py")

if not os.path.exists(pyk2):
    print(f"pykinect2 not found at {pyk2}, skipping patch")
    sys.exit(0)

text = open(pyk2, "r").read()
patched = False

# Fix struct size assertion for 64-bit
if "assert sizeof(tagSTATSTG) == 72" in text:
    text = text.replace(
        "assert sizeof(tagSTATSTG) == 72",
        "assert sizeof(tagSTATSTG) in (72, 80)"
    )
    patched = True

# Fix comtypes version check
if "\nfrom comtypes import _check_version; _check_version" in text:
    text = text.replace(
        "\nfrom comtypes import _check_version; _check_version",
        "\n# from comtypes import _check_version; _check_version"
    )
    patched = True

if patched:
    open(pyk2, "w").write(text)
    print("pykinect2 patched successfully.")
else:
    print("pykinect2 already patched.")

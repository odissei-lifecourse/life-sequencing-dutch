"""Define constants portable across systems.

NOTE: socket.gethostname() does not work as expected on the Snellius login node.
"""

import os
import socket
from pathlib import Path

host_name = socket.gethostname()
os_user = os.environ["USER"]
if "ossc" in host_name:
    OSSC_ROOT = "/gpfs/ostor/ossc9424/homedir/" # legacy; should use os.environ["HOME"] instead
    DATA_ROOT = Path(os.environ["HOME"]) / Path("data/")
    DATA_ROOT = str(DATA_ROOT)
    LOCATION = "ossc"

elif "snellius" in host_name:
    DATA_ROOT = "/projects/0/prjs1019/data/"
    LOCATION = "snellius"








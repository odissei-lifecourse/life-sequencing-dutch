"""Define constants portable across systems.

NOTE: socket.gethostname() does not work as expected on the Snellius login node.
"""

import socket
import os
from pathlib import Path


host_name = socket.gethostname()
os_user = os.environ["USER"]
if "ossc" in host_name: 
    #OSSC_ROOT = "/gpfs/ostor/ossc9424/homedir/" 
     # TODO: this should just be os.environ["HOME"] 
    DATA_ROOT = Path(os.environ["HOME"]) / Path("data/")
    DATA_ROOT = str(DATA_ROOT)
    LOCATION = "ossc"

elif "snellius" in host_name:
    DATA_ROOT = "/projects/0/prjs1019/data/"
    LOCATION = "snellius"








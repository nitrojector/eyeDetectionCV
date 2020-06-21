# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:22:57 2020

@author: marti
"""

import bluetooth, subprocess

print("Performing inquiry...")

nearby_devices = bluetooth.discover_devices(duration=8, lookup_names=True,
                                            flush_cache=True, lookup_class=False)

print("Found {} devices".format(len(nearby_devices)))

# for addr, name in nearby_devices:
#     try:
#         print("{}   {} - {}".format(nearby_devices.index(addr), addr, name))
#     except UnicodeEncodeError:
#         print("{}   {} - {}".format(nearby_devices.index(addr), name.encode("utf-8", "replace")))
    
for index in range(len(nearby_devices)):
  try:
    print("{}   {} - {}".format(index, nearby_devices[index][0], nearby_devices[index][1]))
  except UnicodeEncodeError:
    print("{}   {} - {}".format(index, nearby_devices[index][0], nearby_devices[index][1].encode("utf-8", "replace")))
txt = input("Enter the No. of the device to connect(R for re-scan): ")


deviceNo = int(txt)

name = nearby_devices[deviceNo][1] 
addr = nearby_devices[deviceNo][0]     
port = 1 
passkey = "1234" 

# kill any "bluetooth-agent" process that is already running
subprocess.call("kill -9 `pidof bluetooth-agent`",shell=True)

# Start a new "bluetooth-agent" process where XXXX is the passkey
status = subprocess.call("bluetooth-agent " + passkey + " &",shell=True)

# Now, connect in the same way as always with PyBlueZ

try:
    s = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    s.connect((addr,port))
except bluetooth.btcommon.BluetoothError as err:
    # Error handler
    pass


s.recv(1024) # Buffer size
s.send("t".encode())
import sys
import bluetooth

addr = None

import bluetooth

# print("Performing inquiry...")

# nearby_devices = bluetooth.discover_devices(duration=8, lookup_names=True,
#                                             flush_cache=True, lookup_class=False)

# print("Found {} devices".format(len(nearby_devices)))
        
# for index in range(len(nearby_devices)):
#   try:
#     print("{}   {} - {}".format(index, nearby_devices[index][0], nearby_devices[index][1]))
#   except UnicodeEncodeError:
#     print("{}   {} - {}".format(index, nearby_devices[index][0], nearby_devices[index][1].encode("utf-8", "replace")))

# deviceNo = int(input("Enter the No. of the device to connect"))

# addr = nearby_devices[deviceNo][0]

# Connect to device
if len(sys.argv) < 2:
    print("No device specified. Searching all nearby bluetooth devices for "
          "the SampleServer service...")
else:
    addr = sys.argv[1]
    print("Searching for SampleServer on {}...".format(addr))

# search for the SampleServer service
uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"
service_matches = bluetooth.find_service(uuid=uuid, address=addr)

if len(service_matches) == 0:
    print("Couldn't find the SampleServer service.")
    sys.exit(0)

first_match = service_matches[0]
port = first_match["port"]
name = first_match["name"]
host = first_match["host"]

print("Connecting to \"{}\" on {}".format(name, host))

# Create the client socket
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((host, port))

print("Connected. Type something...")
while True:
    data = input()
    if not data:
        break
    sock.send(data)

sock.close()
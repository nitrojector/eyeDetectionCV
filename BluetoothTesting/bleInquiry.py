import bluetooth

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
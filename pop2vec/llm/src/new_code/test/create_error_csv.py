import re, csv, sys, os

logpath = sys.argv[1]
csv_path = os.path.splitext(os.path.basename(logpath))[0] + ".csv"

with open(logpath) as log, open(csv_path, "w", newline="") as out:
  writer = csv.writer(out)
  writer.writerow(["File Name", "Error Message"])
  for line in log:
    if m := re.search(r"preparing (.+):\n (.+)", line):
      writer.writerow([m.group(1).strip(), m.group(2).strip()])

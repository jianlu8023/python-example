from pathlib import Path
import csv

path = Path("")
lines = path.read_text().splitlines()

reader = csv.reader(lines)

header_now = next(reader)
print(header_now)

import subprocess

subprocess.run(["chmod", "+x", "DistMap2CMapANDsum.py"])

for i in range(1, 21):
    subprocess.run(["./DistMap2CMapANDsum.py", str(i)])
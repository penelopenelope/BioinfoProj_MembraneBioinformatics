import subprocess

subprocess.run(["chmod", "+x", "SumCmap2SVM.py"])

for i in range(1, 21):
    subprocess.run(["./SumCmap2SVM.py", str(i)])
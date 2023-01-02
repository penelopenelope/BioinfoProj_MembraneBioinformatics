import subprocess

subprocess.run(["chmod", "+x", "SplitCmapFiles_alpha_beta.py"])
subprocess.run(["chmod", "+x", "SumCmap2CNN-ModelGeneration.py"])

for i in range(1, 21):
    subprocess.run(["./SplitCmapFiles_alpha_beta.py", str(i)])
    subprocess.run(["./SumCmap2CNN-ModelGeneration.py", str(i)])

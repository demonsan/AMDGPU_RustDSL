import sys
sys.path.insert(0, "tools")
from asm_diff import parse, cls
ins, lab = parse(sys.argv[1])
seg = ins[lab["BB80"]:lab["BB81"]]
runs = []
m = {"mfma.scaled":"Q","mfma.unscaled":"P","mfma.smfmac":"R","mfma.other":"H","ds.read_tr":"t","ds.read":"r","vmem.dma":"D","valu":"v","salu":"s","barrier":"B","setprio":"p","branch":"j","wait":"w","nop":"n"}
for _, t in seg:
    k = m.get(cls(t), "?")
    if runs and runs[-1][0] == k: runs[-1][1] += 1
    else: runs.append([k, 1])
print("TL " + " ".join(f"{k}{n if n > 1 else ''}" for k, n in runs))
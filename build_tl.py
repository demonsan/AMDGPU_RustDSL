import sys, re
sys.path.insert(0, "tools")
from asm_diff import parse, cls
def tl(path, a, b):
    ins, lab = parse(path)
    seg = ins[lab[a]:lab[b]]
    runs = []
    for _, t in seg:
        c = cls(t)
        if c in ("wait", "nop"): c = "w" if c == "wait" else "n"
        short = {"mfma.scaled":"Q","mfma.unscaled":"P","mfma.smfmac":"R","mfma.other":"H","ds.read_tr":"t","ds.read":"r","vmem.dma":"D","valu":"v","salu":"s","barrier":"B","setprio":"p","branch":"j","w":"w","n":"n"}.get(c, "?")
        if runs and runs[-1][0] == short: runs[-1][1] += 1
        else: runs.append([short, 1])
    return " ".join(f"{k}{n if n > 1 else ''}" for k, n in runs)
print("OURS  ", tl(sys.argv[1], "BB80", "BB81"))
print("VENDOR", tl(sys.argv[2], "label_block_A_loop", "label_block_A_drain1"))
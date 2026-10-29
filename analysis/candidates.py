import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np

from fly.check_load import REPO_ROOT, Report
from fly.connectome import CACHE_DIR, ConnectomeNetwork

RESULTS = REPO_ROOT / "results"
SIDES = ("L", "R", "M", "")
NAMED_PREFIX = ("LC10",)
NAMED_EXACT = ("LC11", "LC4", "LPLC2", "LPLC1")
LAMINA = re.compile(r"^(L[1-5]|Lai|Lawf\d|C[23]|T1)$")
MEDULLA = re.compile(r"^(Mi\d+|Tm\d+[a-z]?|TmY\d+[a-z]?|Dm\d+|Pm\d+|Mt\d+)$")


def type_table(net: ConnectomeNetwork, mask: np.ndarray, show_nt: bool = True) -> list[str]:
    idx = np.flatnonzero(mask)
    types = net.columns["type"][idx]
    sides = net.columns["somaSide"][idx]
    nts = net.columns["consensus_nt"][idx]
    by_type: dict[str, Counter] = {}
    nt_of: dict[str, Counter] = {}
    for t, s, nt in zip(types.tolist(), sides.tolist(), nts.tolist()):
        by_type.setdefault(t or "-", Counter())[s] += 1
        nt_of.setdefault(t or "-", Counter())[nt or "-"] += 1
    lines = [f"  {'type':24s}{'L':>6s}{'R':>6s}{'M':>5s}{'none':>6s}{'total':>7s}"
             + ("  transmitter" if show_nt else "")]
    for t in sorted(by_type, key=lambda k: (-sum(by_type[k].values()), k)):
        c = by_type[t]
        row = f"  {t:24s}{c['L']:6d}{c['R']:6d}{c['M']:5d}{c['']:6d}{sum(c.values()):7d}"
        if show_nt:
            row += "  " + ", ".join(f"{n} {k}" for n, k in nt_of[t].most_common())
        lines.append(row)
    lines.append(f"  {len(by_type)} types, {idx.size} neurons")
    return lines


def inputs_report(net: ConnectomeNetwork, rep: Report) -> None:
    sc = net.columns["superclass"]
    cls = net.columns["class"]
    types = net.columns["type"]
    ach = net.columns["consensus_nt"] == "acetylcholine"
    rep.say("== Cholinergic visual_projection types (consensus_nt acetylcholine)")
    for line in type_table(net, (sc == "visual_projection") & ach, show_nt=False):
        rep.say(line)
    rep.say("== Cholinergic optic lobe intrinsic types (superclass ol_intrinsic, acetylcholine)")
    for line in type_table(net, (sc == "ol_intrinsic") & ach, show_nt=False):
        rep.say(line)
    rep.say("== Named candidates, any transmitter")
    named = [(f"{p}*", np.char.startswith(types, p)) for p in NAMED_PREFIX]
    named += [(t, types == t) for t in NAMED_EXACT]
    for label, mask in named:
        rep.say(f"-- {label}")
        for line in type_table(net, mask):
            rep.say(line)
    rep.say("-- lamina types (L1-L5, Lai, Lawf, C2, C3, T1)")
    lamina = np.array([bool(LAMINA.match(t)) for t in types.tolist()])
    for line in type_table(net, lamina):
        rep.say(line)
    rep.say("-- medulla types (Mi, Tm, TmY, Dm, Pm, Mt)")
    medulla = np.array([bool(MEDULLA.match(t)) for t in types.tolist()])
    for line in type_table(net, medulla):
        rep.say(line)
    rep.say("== Mechanosensory class (antennal JO types marked)")
    mech = cls == "mechanosensory"
    for line in type_table(net, mech):
        rep.say(line + ("   <- JO" if line.strip().startswith("JO") else ""))
    rep.say("== ol_sensory superclass")
    for line in type_table(net, sc == "ol_sensory"):
        rep.say(line)


def outputs_report(net: ConnectomeNetwork, rep: Report) -> None:
    sc = net.columns["superclass"]
    rep.say("== Descending neuron types (superclass descending_neuron)")
    for line in type_table(net, sc == "descending_neuron"):
        rep.say(line)
    rep.say("== vnc_motor neurons by type (no nerve column in the cache)")
    for line in type_table(net, sc == "vnc_motor"):
        rep.say(line)
    rep.say("== cb_motor neurons by type")
    for line in type_table(net, sc == "cb_motor"):
        rep.say(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("inputs", "outputs"))
    parser.add_argument("--cache", default=str(CACHE_DIR / "network_A.npz"))
    parser.add_argument("--out")
    args = parser.parse_args()
    net = ConnectomeNetwork(Path(args.cache))
    rep = Report()
    rep.say(f"cache {Path(args.cache).name}, {net.n} neurons, Rule {net.meta['rule']}")
    if args.kind == "inputs":
        inputs_report(net, rep)
    else:
        outputs_report(net, rep)
    out = Path(args.out) if args.out else RESULTS / f"candidate_{args.kind}.txt"
    rep.save(out)


if __name__ == "__main__":
    main()

"""ECG heart-rate monitor: Pan-Tompkins style R-peak detection, heart rate and irregular-beat flags.

Learning project, NOT a medical device.

    python ecg_monitor.py                      # synthetic ECG with known beats, saves results/ecg_demo.png
    python ecg_monitor.py --csv ecg.csv --fs 360
    python ecg_monitor.py --wfdb data/100      # local MIT-BIH record (100.dat/.hea/.atr), needs `pip install wfdb`
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

FS = 360  # Hz, the sampling rate of the MIT-BIH Arrhythmia Database
NORMAL, PREMATURE, PAUSE = 0, 1, 2
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ")  # MIT-BIH annotation symbols that mark a heartbeat

# (amplitude, centre offset in s, width in s) of the Gaussian bumps that make up one beat
_NORMAL_BEAT = [(0.15, -0.20, 0.025), (-0.10, -0.04, 0.010), (1.0, 0.0, 0.012), (-0.20, 0.04, 0.010), (0.30, 0.25, 0.040)]  # P Q R S T
_PVC_BEAT = [(1.2, 0.0, 0.030), (-0.35, 0.22, 0.060)]  # premature beat: wide R, no P wave, inverted T


def synth_ecg(seconds=60, fs=FS, bpm=70, premature_rate=0.08, noise=1.0, seed=0):
    """Synthetic ECG with known beat positions. Returns (ecg, peak_samples, is_premature)."""
    rng = np.random.default_rng(seed)
    rr0 = 60 / bpm
    beats = np.cumsum(rr0 * (1 + 0.03 * rng.normal(size=int(seconds / rr0) + 2)))
    beats = beats[(beats > 1.0) & (beats < seconds - 1.0)]
    early = rng.random(beats.size) < premature_rate
    early[0] = False
    early[1:] &= ~early[:-1]  # never two premature beats in a row
    beats = beats - early * 0.35 * rr0  # arrives 35% early; the next beat stays on schedule (compensatory pause)

    t = np.arange(int(seconds * fs)) / fs
    ecg = np.zeros_like(t)
    for tb, pvc in zip(beats, early):
        lo, hi = int((tb - 0.5) * fs), int((tb + 0.5) * fs)
        seg = t[lo:hi] - tb
        for amp, centre, width in _PVC_BEAT if pvc else _NORMAL_BEAT:
            ecg[lo:hi] += amp * np.exp(-0.5 * ((seg - centre) / width) ** 2)

    wander = 0.3 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi))  # breathing / electrode drift
    mains = 0.1 * np.sin(2 * np.pi * 50 * t)  # powerline interference
    ecg += noise * (wander + mains + 0.05 * rng.normal(size=t.size))
    return ecg, np.round(beats * fs).astype(int), early


def detect_r_peaks(ecg, fs=FS):
    """Band-pass 5-15 Hz, differentiate, square, moving-window integrate, adaptive threshold (Pan-Tompkins).

    shortcut: no search-back for missed beats and no T-wave discrimination; add both if real noisy recordings need them.
    """
    fs = int(round(fs))  # --fs arrives as a float and is used for slicing below
    b, a = butter(2, [5, 15], btype="band", fs=fs)
    x = filtfilt(b, a, ecg)  # zero-phase, so peaks stay where the beats are
    win = int(0.15 * fs)
    integ = np.convolve(np.gradient(x) ** 2, np.ones(win) / win, mode="same")

    cand, _ = find_peaks(integ, distance=int(0.2 * fs))  # 200 ms refractory period
    start = integ[: 2 * fs]
    spki, npki = 0.25 * start.max(), 0.5 * start.mean()  # running signal-peak and noise-peak levels
    peaks = []
    for c in cand:
        if integ[c] > npki + 0.25 * (spki - npki):
            peaks.append(c)
            spki = 0.125 * integ[c] + 0.875 * spki
        else:
            npki = 0.125 * integ[c] + 0.875 * npki

    half = int(0.1 * fs)  # snap each detection to the largest deflection of the filtered signal
    return np.unique([max(0, c - half) + np.argmax(np.abs(x[max(0, c - half): c + half])) for c in peaks])


def analyse(peaks, fs=FS):
    """Mean heart rate plus a per-beat label: premature if its interval is <80% of the recent median, pause if >120%."""
    rr = np.diff(peaks) / fs
    kind = np.zeros(len(peaks), dtype=int)
    for k in range(4, len(rr) + 1):  # rr[k-1] is the interval that ends at beat k; need 3+ earlier intervals
        ref = np.median(rr[max(0, k - 9): k - 1])
        if rr[k - 1] < 0.8 * ref:
            kind[k] = PREMATURE
        elif rr[k - 1] > 1.2 * ref:
            kind[k] = PAUSE
    return {"rr": rr, "bpm": 60 / rr.mean() if rr.size else float("nan"), "kind": kind}


def score(true, det, fs=FS, tol=0.075):
    """Match detections to true beats within +-75 ms. Returns (tp, fn, fp, {true_index: detection_index})."""
    det = np.asarray(det)
    used = np.zeros(det.size, dtype=bool)
    matches = {}
    for i, t in enumerate(true):
        if det.size == 0:
            break
        dist = np.abs(det - t).astype(float)
        dist[used] = np.inf
        j = int(dist.argmin())
        if dist[j] <= tol * fs:
            used[j] = True
            matches[i] = j
    tp = len(matches)
    return tp, len(true) - tp, det.size - tp, matches


def noise_sweep(levels=(0.5, 1, 2, 3, 4, 6, 8), seeds=range(5)):
    """Sensitivity and precision of the detector as the noise level rises (synthetic data only)."""
    sens, prec = [], []
    for lv in levels:
        s, p = [], []
        for sd in seeds:
            ecg, truth, _ = synth_ecg(noise=lv, seed=sd)
            det = detect_r_peaks(ecg)
            tp, fn, fp, _ = score(truth, det)
            s.append(tp / len(truth))
            p.append(tp / max(1, tp + fp))
        sens.append(np.mean(s))
        prec.append(np.mean(p))
    return list(levels), sens, prec


def load_wfdb(path):
    """Local MIT-BIH style record -> (signal, fs, true beat samples)."""
    import wfdb  # optional dependency

    rec, ann = wfdb.rdrecord(path), wfdb.rdann(path, "atr")
    beats = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS])
    return rec.p_signal[:, 0], rec.fs, beats


def plot(ecg, fs, peaks, res, sweep, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, AQUA, VIOLET, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#0b0b0b", "#52514e", "#fcfcfb"
    rows = 3 if sweep else 2
    fig, axes = plt.subplots(rows, 1, figsize=(8, 3.1 * rows), facecolor=SURF)
    t = np.arange(len(ecg)) / fs
    kind = res["kind"]

    ax = axes[0]
    m = t < 10
    ax.plot(t[m], ecg[m], color=BLUE, lw=1, label="ECG (noisy)")
    for k, colour, marker, name in ((NORMAL, AQUA, "o", "normal beat"), (PREMATURE, ORANGE, "D", "premature beat"), (PAUSE, VIOLET, "s", "pause")):
        sel = peaks[(kind == k) & (peaks / fs < 10)]
        if sel.size:
            ax.plot(sel / fs, ecg[sel], marker, color=colour, ms=7, mec=SURF, mew=1.5, ls="none", label=name)
    ax.set(xlabel="Time (s)", ylabel="Amplitude (a.u.)", title="Detected beats, first 10 seconds")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28), labelcolor=INK)

    ax = axes[1]
    ax.step(peaks[1:] / fs, 60 / res["rr"], where="post", color=BLUE, lw=2, label="heart rate")
    flagged = kind[1:] != NORMAL
    ax.plot(peaks[1:][flagged] / fs, (60 / res["rr"])[flagged], "D", color=ORANGE, ms=7, mec=SURF, mew=1.5, ls="none", label="flagged beat")
    ax.set(xlabel="Time (s)", ylabel="Beats per minute", title=f"Heart rate per beat (mean {res['bpm']:.1f} bpm)")
    ax.legend(frameon=False, labelcolor=INK)

    if sweep:
        levels, sens, prec = sweep
        ax = axes[2]
        ax.plot(levels, sens, "o-", color=BLUE, lw=2, ms=8, mec=SURF, mew=2, label="sensitivity")
        ax.plot(levels, prec, "s-", color=ORANGE, lw=2, ms=8, mec=SURF, mew=2, label="precision")
        ax.set(ylim=(0, 1.05), xlabel="Noise level (1 = default: breathing drift + 50 Hz mains + random noise)", ylabel="Fraction of beats", title="Accuracy as noise increases (synthetic ECG, 5 runs per point)")
        ax.legend(frameon=False, labelcolor=INK, loc="lower left")

    for ax in axes:
        ax.set_facecolor(SURF)
        ax.grid(True, color="#e4e3df", lw=0.8)
        ax.tick_params(colors=MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        title = ax.get_title()
        ax.set_title("")
        ax.set_title(title, loc="left", color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    Path(out).parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="text file with one ECG sample per line")
    ap.add_argument("--wfdb", help="path to a local MIT-BIH record without extension, e.g. data/100")
    ap.add_argument("--fs", type=float, default=FS, help="sampling rate of --csv in Hz")
    args = ap.parse_args()

    truth = sweep = None
    if args.wfdb:
        ecg, fs, truth = load_wfdb(args.wfdb)
    elif args.csv:
        ecg, fs = np.loadtxt(args.csv), args.fs
    else:
        ecg, truth, _ = synth_ecg()
        fs, sweep = FS, noise_sweep()

    peaks = detect_r_peaks(ecg, fs)
    res = analyse(peaks, fs)
    counts = np.bincount(res["kind"], minlength=3)
    print(f"{len(peaks)} beats, mean heart rate {res['bpm']:.1f} bpm, "
          f"{counts[PREMATURE]} premature, {counts[PAUSE]} pauses flagged")
    if truth is not None:
        tp, fn, fp, _ = score(truth, peaks, fs)
        print(f"vs reference beats: sensitivity {tp / len(truth):.3f}, precision {tp / max(1, tp + fp):.3f} (missed {fn}, extra {fp})")
    own_data = args.csv or args.wfdb  # keep the demo chart the README shows from being overwritten
    out = Path(__file__).parent / "results" / ("ecg_custom.png" if own_data else "ecg_demo.png")
    plot(ecg, fs, peaks, res, sweep, out)
    print("saved", out)


if __name__ == "__main__":
    main()

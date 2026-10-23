"""Run with `python -m pytest -q`."""
import numpy as np

from ecg_monitor import FS, PREMATURE, analyse, detect_r_peaks, score, synth_ecg

SEEDS = range(5)


def test_finds_nearly_every_beat_at_default_noise():
    for seed in SEEDS:
        ecg, truth, _ = synth_ecg(seed=seed)
        tp, fn, fp, _ = score(truth, detect_r_peaks(ecg))
        assert tp / len(truth) >= 0.99, (seed, tp, fn)
        assert fp <= 1, (seed, fp)


def test_heart_rate_within_2_bpm():
    for seed in SEEDS:
        ecg, truth, _ = synth_ecg(seed=seed)
        true_bpm = 60 / (np.diff(truth) / FS).mean()
        assert abs(analyse(detect_r_peaks(ecg))["bpm"] - true_bpm) < 2


def test_flags_premature_beats_and_not_regular_ones():
    hit = total = false_flags = 0
    for seed in SEEDS:
        ecg, truth, early = synth_ecg(premature_rate=0.15, seed=seed)
        peaks = detect_r_peaks(ecg)
        kind = analyse(peaks)["kind"]
        _, _, _, matches = score(truth, peaks)
        for i, j in matches.items():
            if early[i]:
                total += 1
                hit += kind[j] == PREMATURE
            elif kind[j] == PREMATURE:
                false_flags += 1
    assert total > 10 and hit / total >= 0.9, (hit, total)
    assert false_flags <= 2, false_flags


def test_no_premature_flags_on_a_regular_rhythm():
    ecg, _, _ = synth_ecg(premature_rate=0, seed=1)
    assert (analyse(detect_r_peaks(ecg))["kind"] == PREMATURE).sum() == 0


def test_sampling_rate_given_as_float_works():  # `--fs 360` on the command line arrives as 360.0
    ecg, _, _ = synth_ecg(seconds=20, seed=2)
    assert (detect_r_peaks(ecg, 360.0) == detect_r_peaks(ecg, 360)).all()


def test_score_counts_hits_misses_and_extras():
    tp, fn, fp, _ = score([100, 500, 900], [102, 898, 2000], fs=360)
    assert (tp, fn, fp) == (2, 1, 1)

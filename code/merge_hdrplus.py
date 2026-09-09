"""Comparison F: the noise-variance frequency-domain merge used by HDR+ style bursts.

Transcribed from the published merge rule: per overlapping tile, for each alternate frame
z, with D_z = T_0 - T_z in the frequency domain,

    A_z(w) = |D_z(w)|^2 / (|D_z(w)|^2 + c * sigma^2)
    merged(w) = (1/N) * sum_z [ T_z(w) + A_z(w) * (T_0(w) - T_z(w)) ]

so a frequency where the frames disagree far more than the noise level falls back to the
reference tile, and a frequency where they agree is averaged. sigma is estimated from the
frames themselves rather than assumed. This is the opponent our proposal has to beat; the
prior survey was explicit that beating a plain average proves nothing.

No alignment stage: the benchmark's short frames are tripod-static captures of the same
scene, which is a property of the data and is stated as such wherever this is reported.
"""
import numpy as np

TILE, C_DEFAULT = 16, 8.0


def _hann2(n):
    w = np.hanning(n + 1)[:-1]
    return np.outer(w, w)


def estimate_sigma(frames):
    """noise std per channel from frame differences of a static scene"""
    d = np.diff(np.asarray(frames, np.float64), axis=0)
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2.0))


def merge(frames, c=C_DEFAULT, tile=TILE, sigma=None):
    """frames: list of (H,W,3) float arrays in [0,1]; frames[0] is the reference."""
    F = np.asarray(frames, np.float64)
    N, H, W, C = F.shape
    if N == 1:
        return F[0].copy()
    if sigma is None:
        sigma = estimate_sigma(F)
    sigma = max(float(sigma), 1e-6)                 # identical frames must not divide by zero
    s2 = (c * sigma ** 2) * (tile * tile)          # Parseval: per-coefficient noise power
    win = _hann2(tile)
    step = tile // 2
    # reflect-pad by half a tile so the Hann overlap-add sums to one at the borders too;
    # without this the edge pixels sit under a near-zero window and blow up on division
    F = np.pad(F, ((0, 0), (step, step), (step, step), (0, 0)), mode="reflect")
    H, W = F.shape[1], F.shape[2]
    out = np.zeros((H, W, C)); wsum = np.zeros((H, W, 1))
    ys = list(range(0, H - tile + 1, step)) + ([H - tile] if (H - tile) % step else [])
    xs = list(range(0, W - tile + 1, step)) + ([W - tile] if (W - tile) % step else [])
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            blk = F[:, y:y + tile, x:x + tile, :] * win[None, :, :, None]
            T = np.fft.fft2(blk, axes=(1, 2))
            T0 = T[0]
            acc = np.zeros_like(T0)
            for z in range(N):
                D = T0 - T[z]
                A = (np.abs(D) ** 2) / (np.abs(D) ** 2 + s2)
                acc += T[z] + A * D
            m = np.real(np.fft.ifft2(acc / N, axes=(0, 1)))
            out[y:y + tile, x:x + tile, :] += m
            wsum[y:y + tile, x:x + tile, 0] += win
    out = out / np.maximum(wsum, 1e-8)
    return out[step:-step, step:-step, :]

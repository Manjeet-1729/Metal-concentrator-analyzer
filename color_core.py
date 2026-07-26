"""
color_core.py

Single source of truth for the colour-extraction / matching pipeline used by
BOTH the Streamlit app (app.py) and the calibration-data extraction script
(extract_calibration_rgb.py). Keeping this logic in one place guarantees the
CSV reference data and the live app are always computed the exact same way —
no drift, no accidental mismatch.
"""
import numpy as np
import pandas as pd
from PIL import Image


# ── Colour-space helpers (vectorised, no extra dependencies) ───────────────────
def rgb_to_hsv_np(pixels: np.ndarray) -> np.ndarray:
    """Vectorised RGB (0-255) -> HSV (H:0-360, S:0-1, V:0-1) for an (N,3) array."""
    p = pixels.astype(np.float64) / 255.0
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    maxc = np.max(p, axis=1)
    minc = np.min(p, axis=1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc == 0, 0, delta / np.where(maxc == 0, 1, maxc))

    h = np.zeros_like(maxc)
    mask = delta != 0
    rc = np.zeros_like(maxc); gc = np.zeros_like(maxc); bc = np.zeros_like(maxc)
    safe_delta = np.where(delta == 0, 1, delta)
    rc[mask] = ((maxc - r) / safe_delta)[mask]
    gc[mask] = ((maxc - g) / safe_delta)[mask]
    bc[mask] = ((maxc - b) / safe_delta)[mask]

    is_r = mask & (maxc == r)
    is_g = mask & (maxc == g) & ~is_r
    is_b = mask & (maxc == b) & ~is_r & ~is_g

    h[is_r] = (bc - gc)[is_r]
    h[is_g] = 2.0 + (rc - bc)[is_g]
    h[is_b] = 4.0 + (gc - rc)[is_b]
    h = (h / 6.0) % 1.0
    return np.stack([h * 360, s, v], axis=1)


def laplacian_variance(gray: np.ndarray) -> float:
    """Blur metric: variance of a 3x3 Laplacian response (higher = sharper)."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    padded = np.pad(gray.astype(np.float64), 1, mode="edge")
    h, w = gray.shape
    conv = np.zeros((h, w), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if kernel[i, j] != 0:
                conv += kernel[i, j] * padded[i:i + h, j:j + w]
    return float(conv.var())


def gray_world_gains(pixels: np.ndarray) -> np.ndarray:
    """Estimate per-channel correction gains assuming the scene averages to neutral grey."""
    means = pixels.astype(np.float64).mean(axis=0)
    means = np.clip(means, 1, None)
    overall = means.mean()
    gains = overall / means
    return np.clip(gains, 0.6, 1.6)  # avoid over-correcting extreme scenes


# ── Core extraction ──────────────────────────────────────────────────────────
def estimate_lighting_gains(full: np.ndarray, region_frac=(0.10, 0.90, 0.10, 0.90)) -> np.ndarray:
    """
    Estimates gray-world correction gains from a region of a full RGB array.
    Kept separate from extract_liquid_color() so callers with extra context
    (e.g. a whole multi-tube strip photo) can estimate lighting ONCE from a
    region with real background, then apply it consistently to sub-crops that
    don't have enough neutral background of their own to estimate lighting from.
    """
    h, w, _ = full.shape
    fx0, fx1, fy0, fy1 = region_frac
    x0, x1 = int(w * fx0), int(w * fx1)
    y0, y1 = int(h * fy0), int(h * fy1)
    region = full[y0:y1, x0:x1].reshape(-1, 3)
    return gray_world_gains(region)


def extract_liquid_color(rgb_array: np.ndarray, gains: np.ndarray,
                          search_frac=(0.20, 0.80, 0.15, 0.85),
                          sat_thresh: float = 0.15, val_min: float = 0.08, val_max: float = 0.95,
                          min_pixel_frac: float = 0.02, min_pixel_count: int = 30,
                          dark_thresh: float = 35, bright_thresh: float = 240,
                          glare_thresh_pct: float = 12, blur_thresh: float = 8) -> dict:
    """
    Given an already-loaded RGB uint8/float array and known lighting-correction
    gains, detects the liquid region within search_frac bounds (fx0, fx1, fy0, fy1)
    and returns the extracted colour + quality diagnostics. This is the shared
    core used by analyse_image() and the calibration script. Thresholds are
    exposed as parameters so callers with different photo styles (full ambient
    photos vs. tightly pre-cropped capless tube photos) can tune sensitivity
    without duplicating the underlying logic.
    """
    h, w, _ = rgb_array.shape
    corrected_full = np.clip(rgb_array.astype(np.float64) * gains, 0, 255)

    fx0, fx1, fy0, fy1 = search_frac
    sx0, sx1 = int(w * fx0), int(w * fx1)
    sy0, sy1 = int(h * fy0), int(h * fy1)
    search_crop = corrected_full[sy0:sy1, sx0:sx1]
    search_pixels = search_crop.reshape(-1, 3)

    hsv = rgb_to_hsv_np(search_pixels)
    sat, val = hsv[:, 1], hsv[:, 2]

    # "liquid-like" = has real colour (not glass/background/glare/shadow).
    # Note: this does NOT require a light background — a saturated colour against a
    # dark/black background (e.g. backlit tube photos) is detected just as well, since
    # the black background simply has very low V and gets excluded by the val_min test.
    liquid_mask = (sat > sat_thresh) & (val > val_min) & (val < val_max)
    glare_mask = (val > 0.95) & (sat < sat_thresh)
    glare_pct = float(glare_mask.mean() * 100)

    flags = []  # list of (severity, message)  severity: "bad" | "mid"
    region_method = "auto-detected"
    if liquid_mask.sum() >= max(min_pixel_count, min_pixel_frac * len(search_pixels)):
        liquid_pixels = search_pixels[liquid_mask]
    else:
        # Fallback: narrow centre crop, old-style brightness filter
        region_method = "fallback (fixed centre crop)"
        cx0 = sx0 + int((sx1 - sx0) * 0.25)
        cx1 = sx0 + int((sx1 - sx0) * 0.75)
        cy0 = sy0 + int((sy1 - sy0) * 0.25)
        cy1 = sy0 + int((sy1 - sy0) * 0.75)
        fallback_pixels = corrected_full[cy0:cy1, cx0:cx1].reshape(-1, 3)
        fb_brightness = fallback_pixels.mean(axis=1)
        fb_mask = (fb_brightness > 15) & (fb_brightness < 240)
        liquid_pixels = fallback_pixels[fb_mask] if fb_mask.sum() > 10 else fallback_pixels
        flags.append(("mid", "Couldn't confidently detect the tube/liquid. Make sure the photo is cropped tightly to the tube body."))

    if glare_pct > glare_thresh_pct:
        flags.append(("mid", f"Glare detected on ~{glare_pct:.0f}% of the tube area. Avoid direct reflections/flash."))

    # ---- Quality checks computed from the LIQUID REGION itself, not the whole frame ------
    # (a photo with a deliberately dark/black background around a well-lit tube should not
    # be flagged as "too dark" just because most of the frame is black background)
    region_brightness = float(liquid_pixels.mean())
    if region_brightness < dark_thresh:
        flags.append(("bad", "The liquid/tube area itself looks too dark. Increase lighting on the sample (background can stay dark)."))
    elif region_brightness > bright_thresh:
        flags.append(("mid", "The liquid/tube area looks overexposed/washed out. Reduce direct light or flash on the tube."))

    search_gray = search_crop.mean(axis=2)
    step = max(1, max(search_gray.shape) // 300)
    blur_score = laplacian_variance(search_gray[::step, ::step])
    if blur_score < blur_thresh:
        flags.append(("bad", "Photo appears blurry. Hold the camera steady and refocus."))

    r = int(np.median(liquid_pixels[:, 0]))
    g = int(np.median(liquid_pixels[:, 1]))
    b = int(np.median(liquid_pixels[:, 2]))

    return {
        "rgb": (r, g, b),
        "crop_box": (sx0, sy0, sx1, sy1),
        "region_method": region_method,
        "flags": flags,
        "brightness": region_brightness,
        "blur_score": blur_score,
        "glare_pct": glare_pct,
    }


def analyse_image(img: Image.Image) -> dict:
    """
    Canonical extraction function for the CURRENT expected input format:
    a photo that is ALREADY tightly cropped to just the test-tube body,
    with the cap excluded and minimal surrounding background (see the
    in-app photo guidelines).

    Deliberately applies NO gray-world / lighting correction. Earlier testing
    showed that when a crop is dominated by the liquid's own saturated colour
    with little-to-no neutral background (which is exactly what a tight
    tube-only crop looks like), gray-world wrongly treats that colour as a
    lighting cast and desaturates it — actively hurting accuracy. Since these
    photos are expected to come from a consistent capture setup, raw pixel
    values are matched directly against a reference CSV built the same way
    (see extract_calibration_rgb.py), which is more accurate than a per-image
    "correction" with nothing reliable to calibrate itself against.
    """
    img_rgb = img.convert("RGB")
    full = np.array(img_rgb)
    identity_gains = np.array([1.0, 1.0, 1.0])
    # Nearly full-frame search (small margin to avoid edge/compression artifacts),
    # since the input is already expected to be a tight tube-only crop.
    return extract_liquid_color(
        full, identity_gains,
        search_frac=(0.03, 0.97, 0.02, 0.98),
        sat_thresh=0.12, val_min=0.06, val_max=0.97,
        min_pixel_frac=0.02, min_pixel_count=20,
        dark_thresh=30, bright_thresh=245,
        glare_thresh_pct=15, blur_thresh=5,
    )


# ── Matching with interpolation ─────────────────────────────────────────────
def find_match(df: pd.DataFrame, r: int, g: int, b: int, conc_col: str):
    """
    Finds the nearest 2 reference rows by RGB distance and returns a
    distance-weighted interpolated concentration, plus the single nearest
    row (for display) and the raw nearest distance (for confidence).
    """
    diffs = np.sqrt((df["R"] - r) ** 2 + (df["G"] - g) ** 2 + (df["B"] - b) ** 2).to_numpy()
    order = np.argsort(diffs)
    nearest_idx = order[0]
    nearest_row = df.iloc[nearest_idx]
    nearest_dist = float(diffs[nearest_idx])

    if len(order) >= 2:
        i1, i2 = order[0], order[1]
        d1, d2 = diffs[i1], diffs[i2]
        if d1 + d2 == 0:
            interp_conc = float(df.iloc[i1][conc_col])
        else:
            # inverse-distance weighting (avoid div-by-zero)
            w1 = 1.0 / max(d1, 1e-6)
            w2 = 1.0 / max(d2, 1e-6)
            interp_conc = float((df.iloc[i1][conc_col] * w1 + df.iloc[i2][conc_col] * w2) / (w1 + w2))
    else:
        interp_conc = float(nearest_row[conc_col])

    return {
        "interp_conc": interp_conc,
        "nearest_row": nearest_row,
        "nearest_dist": nearest_dist,
    }


def match_confidence(dist: float, df: pd.DataFrame):
    """Scales confidence relative to the spread of distances actually seen across the reference set,
    so the thresholds adapt to each CSV instead of using arbitrary fixed numbers."""
    pts = df[["R", "G", "B"]].to_numpy()
    if len(pts) > 1:
        sample = pts[np.random.choice(len(pts), min(len(pts), 40), replace=False)]
        internal_dists = []
        for p in sample:
            d = np.sqrt(((pts - p) ** 2).sum(axis=1))
            d = d[d > 0]
            if len(d):
                internal_dists.append(d.min())
        scale = float(np.median(internal_dists)) if internal_dists else 20.0
        scale = max(scale, 8.0)
    else:
        scale = 20.0

    if dist < scale:
        return "High", "#2e7d32", 90
    elif dist < scale * 3:
        return "Medium", "#e65100", 60
    else:
        return "Low", "#c62828", 25
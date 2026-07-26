import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Metal Concentration Analyzer",
    page_icon="🧪",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #f0f4f8; }
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    }
    .hero h1 { color: #e0f0ff; font-size: 2.1rem; margin: 0 0 .4rem; }
    .hero p  { color: #90caf9; font-size: 1rem; margin: 0; }
    .test-card {
        background: white;
        border: 2px solid #e3eaf5;
        border-radius: 14px;
        padding: 1.6rem 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-top: .5rem;
    }
    .test-card.selected { border-color: #1976d2; background: #e3f2fd; }
    .test-card .icon { font-size: 2.8rem; margin-bottom: .5rem; }
    .test-card h3  { color: #1a237e; margin: 0 0 .3rem; }
    .test-card p   { color: #546e7a; font-size: .85rem; margin: 0; }
    .result-box {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        border: 2px solid #66bb6a;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-box.low-conf {
        background: linear-gradient(135deg, #fff3e0, #fff8e1);
        border: 2px solid #ffa726;
    }
    .result-box h2 { color: #2e7d32; font-size: 2.4rem; margin: 0; }
    .result-box.low-conf h2 { color: #e65100; }
    .result-box p  { color: #388e3c; margin: .4rem 0 0; font-size: 1rem; }
    .result-box.low-conf p { color: #ef6c00; }
    .rgb-swatch {
        display: inline-block;
        border-radius: 8px;
        padding: .4rem 1.1rem;
        font-size: 1rem;
        font-weight: 600;
        margin: .5rem auto;
        border: 2px solid rgba(0,0,0,0.12);
    }
    .step-label {
        background: #1976d2;
        color: white;
        border-radius: 20px;
        padding: .25rem .9rem;
        font-size: .8rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: .6rem;
        letter-spacing: .05em;
    }
    .crop-box {
        border: 3px dashed #1976d2;
        border-radius: 10px;
        padding: .8rem;
        background: #e3f2fd;
        text-align: center;
        font-size: .85rem;
        color: #1565c0;
        margin-top: .5rem;
    }
    .warn-box {
        border-radius: 10px;
        padding: .7rem 1rem;
        margin-top: .5rem;
        font-size: .87rem;
    }
    .warn-bad { background: #ffebee; border: 1.5px solid #ef5350; color: #b71c1c; }
    .warn-mid { background: #fff8e1; border: 1.5px solid #ffca28; color: #8d6e00; }
    .conf-pill {
        display: inline-block;
        border-radius: 20px;
        padding: .3rem 1rem;
        font-weight: 700;
        font-size: .85rem;
        margin: .2rem .3rem;
    }
    .guide-box {
        background: #fff8e1;
        border: 2px solid #ffca28;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin: .6rem 0 1rem;
    }
    .guide-title {
        font-weight: 700;
        color: #8d6e00;
        font-size: 1.02rem;
        margin: 0 0 .6rem;
    }
    .guide-table { width: 100%; border-collapse: collapse; }
    .guide-table td {
        padding: .35rem .5rem;
        font-size: .87rem;
        color: #5d4a00;
        vertical-align: top;
    }
    .guide-table td:first-child { width: 2rem; font-size: 1.1rem; text-align: center; }

    /* Defensive overrides: force readable text on native Streamlit widgets
       (checkbox labels, expander headers, plain markdown/write) in case the
       viewer's browser/OS dark-mode preference leaks through despite the
       locked light theme in .streamlit/config.toml */
    [data-testid="stCheckbox"] label p,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stMarkdownContainer"] p,
    .stAlert p {
        color: #1a1a2e !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧪 Metal Concentration Analyzer</h1>
  <p>Upload a test-tube image → colour extracted from the liquid → matched to concentration</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "test_type" not in st.session_state:
    st.session_state.test_type = None

# ── Load CSVs ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path)
        required = {"R", "G", "B"}
        if not required.issubset(df.columns):
            st.error(f"{path} is missing required columns {required}")
            return None
        df = df.dropna(subset=["R", "G", "B"]).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Could not load {path}: {e}")
        return None

al_df = load_csv("al_data.csv")
mn_df = load_csv("mn_data.csv")

# ── Shared extraction/matching logic (single source of truth) ─────────────────
from color_core import rgb_to_hsv_np, laplacian_variance, gray_world_gains, analyse_image, find_match, match_confidence


# ── Step 1 — Choose test type ──────────────────────────────────────────────────
st.markdown('<span class="step-label">STEP 1 — Choose Test Type</span>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    al_sel = st.session_state.test_type == "Al"
    if st.button("🔵  Aluminium (Al) Test", width="stretch",
                 type="primary" if al_sel else "secondary"):
        st.session_state.test_type = "Al"
        st.rerun()
    st.markdown(f"""
    <div class="test-card {'selected' if al_sel else ''}">
      <div class="icon">🔵</div>
      <h3>Aluminium Test</h3>
      <p>Detects Al³⁺ ions.<br>Solution turns blue-teal.</p>
    </div>""", unsafe_allow_html=True)

with col2:
    mn_sel = st.session_state.test_type == "Mn"
    if st.button("🩷  Manganese (Mn) Test", width="stretch",
                 type="primary" if mn_sel else "secondary"):
        st.session_state.test_type = "Mn"
        st.rerun()
    st.markdown(f"""
    <div class="test-card {'selected' if mn_sel else ''}">
      <div class="icon">🩷</div>
      <h3>Manganese Test</h3>
      <p>Detects Mn²⁺ ions.<br>Solution turns orange-pink.</p>
    </div>""", unsafe_allow_html=True)

# ── Step 2 — Upload ────────────────────────────────────────────────────────────
if st.session_state.test_type:
    test     = st.session_state.test_type
    df       = al_df if test == "Al" else mn_df
    conc_col = "Al_concentration_ppm" if test == "Al" else "Mn_concentration_ppm"
    element  = "Aluminium (Al)"       if test == "Al" else "Manganese (Mn)"

    st.markdown("---")
    st.markdown(f'<span class="step-label">STEP 2 — Upload Test Tube Image ({element})</span>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-box">
      <p class="guide-title">📸 How to prepare your photo for accurate results</p>
      <table class="guide-table">
        <tr><td>✂️</td><td><strong>Crop tightly to the tube body</strong> — the photo should show just the test tube itself, filling almost the entire frame.</td></tr>
        <tr><td>🚫</td><td><strong>Exclude the cap</strong> — crop it out so the top of the photo starts at the liquid, not the cap/lid.</td></tr>
        <tr><td>🚫</td><td><strong>No glare</strong> — avoid a bright reflection sitting directly on the liquid.</td></tr>
        <tr><td>🖐️</td><td><strong>Steady, in-focus shot</strong> — avoid motion blur.</td></tr>
        <tr><td>🕐</td><td><strong>Timing</strong> — photograph immediately after the recommended reaction time, not before/after (colour shifts with time).</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

    photo_confirmed = st.checkbox(
        "✅ My photo is cropped tightly to just the tube body, with the cap excluded.",
        key="photo_guidelines_ack"
    )

    if not photo_confirmed:
        st.warning("Please confirm your photo is cropped correctly before uploading — this is the single biggest factor in getting an accurate result.")

    uploaded_img = st.file_uploader(
        "Upload test tube photo (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="tube_image",
        disabled=not photo_confirmed
    )

    if uploaded_img:
        img_bytes = uploaded_img.read()
        img = Image.open(io.BytesIO(img_bytes))

        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img, caption="Uploaded image", width="stretch")
        with c2:
            st.markdown("**Image info**")
            st.write(f"• Size: {img.size[0]} × {img.size[1]} px")
            st.write(f"• Mode: {img.mode}")
            st.write(f"• File: {uploaded_img.name}")
            st.markdown('<div class="crop-box">📐 The colour is sampled across the whole photo — since it should already be cropped tightly to just the tube body (no cap), no further region-guessing is needed.</div>',
                        unsafe_allow_html=True)

        # ── Step 3 — Analyse ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<span class="step-label">STEP 3 — Analyse</span>', unsafe_allow_html=True)

        if df is None:
            st.error("❌ Could not load concentration data. Check your CSV files.")
        else:
            if st.button("🔬 Analyse Concentration", type="primary", width="stretch"):
                with st.spinner("Checking image quality and extracting colour…"):
                    result = analyse_image(img)

                r, g, b = result["rgb"]
                x0, y0, x1, y1 = result["crop_box"]
                img_crop = img.convert("RGB").crop((x0, y0, x1, y1))
                hex_col  = f"#{r:02x}{g:02x}{b:02x}"
                lum      = 0.299 * r + 0.587 * g + 0.114 * b
                txt_col  = "#000" if lum > 128 else "#fff"

                # ---- Quality warnings, shown first ----
                if result["flags"]:
                    for severity, msg in result["flags"]:
                        css_class = "warn-bad" if severity == "bad" else "warn-mid"
                        icon = "🚫" if severity == "bad" else "⚠️"
                        st.markdown(f'<div class="warn-box {css_class}">{icon} {msg}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warn-box" style="background:#e8f5e9;border:1.5px solid #66bb6a;color:#1b5e20;">✅ Image quality looks good.</div>', unsafe_allow_html=True)

                ca, cb = st.columns([1, 2])
                with ca:
                    st.image(img_crop, caption="Liquid pixels detected within this crop", width="stretch")
                with cb:
                    st.markdown(f"""
                    <div style="text-align:center; padding:1rem;">
                      <p style="margin:0; color:#555; font-size:.9rem;">🎨 Extracted solution colour</p>
                      <span class="rgb-swatch" style="background:{hex_col}; color:{txt_col};">
                        RGB ({r}, {g}, {b}) &nbsp;|&nbsp; {hex_col}
                      </span>
                      <p style="color:#888; font-size:.8rem; margin:.4rem 0 0;">
                        Detection method: {result['region_method']}
                      </p>
                    </div>
                    """, unsafe_allow_html=True)

                # ---- Matching ----
                match = find_match(df, r, g, b, conc_col)
                distance = match["nearest_dist"]
                nearest_row = match["nearest_row"]
                interp_conc = match["interp_conc"]

                matched_rgb = (int(nearest_row["R"]), int(nearest_row["G"]), int(nearest_row["B"]))
                matched_hex = "#{:02x}{:02x}{:02x}".format(*matched_rgb)
                matched_lum = 0.299 * matched_rgb[0] + 0.587 * matched_rgb[1] + 0.114 * matched_rgb[2]
                matched_txt = "#000" if matched_lum > 128 else "#fff"

                conf_label, conf_color, conf_score = match_confidence(distance, df)
                low_conf = conf_label == "Low"

                conc_label = f"{interp_conc:.1f} µM" if interp_conc != 0 else "Blank (0 µM)"
                box_class = "result-box low-conf" if low_conf else "result-box"
                headline = "⚠️ Best available match (low confidence)" if low_conf else "✅ Closest match found"

                st.markdown(f"""
                <div class="{box_class}">
                  <p style="font-size:.9rem; margin-bottom:.3rem;">{headline}</p>
                  <h2>{conc_label}</h2>
                  <p>{element} concentration (interpolated between 2 nearest reference points)</p>
                  <hr style="border:1px solid #c8e6c9; margin:1rem 0;">
                  <p style="margin:0; font-size:.85rem;">
                    Nearest single reference:&nbsp;
                    <span class="rgb-swatch" style="background:{matched_hex}; color:{matched_txt}; font-size:.8rem; padding:.2rem .7rem;">
                      RGB ({matched_rgb[0]}, {matched_rgb[1]}, {matched_rgb[2]}) → {nearest_row[conc_col]} µM
                    </span>
                    &nbsp;| Distance: <strong>{distance:.1f}</strong>
                  </p>
                </div>
                """, unsafe_allow_html=True)

                if low_conf:
                    st.markdown(
                        '<div class="warn-box warn-bad">🚫 This colour doesn\'t closely resemble any calibrated reference. '
                        'The value above is only the *closest available* match and may be inaccurate. '
                        'Please retake the photo with better lighting, less glare, and the tube filling more of the frame.</div>',
                        unsafe_allow_html=True
                    )

                # ---- Confidence pills ----
                quality_score = 100
                for severity, _ in result["flags"]:
                    quality_score -= 35 if severity == "bad" else 15
                quality_score = max(0, quality_score)
                q_color = "#2e7d32" if quality_score >= 70 else ("#e65100" if quality_score >= 40 else "#c62828")

                st.markdown(f"""
                <p style="text-align:center; margin-top:.9rem;">
                  <span class="conf-pill" style="background:#e8f5e9; color:{q_color};">📷 Image Quality: {quality_score}/100</span>
                  <span class="conf-pill" style="background:#e3f2fd; color:{conf_color};">🎯 Match Confidence: {conf_label} ({conf_score}/100)</span>
                </p>
                """, unsafe_allow_html=True)

                with st.expander("📋 View full reference data"):
                    st.dataframe(df, width="stretch")

                with st.expander("🔍 Diagnostic details"):
                    st.write(f"- Liquid-region brightness (0-255): {result['brightness']:.1f}")
                    st.write(f"- Blur score (higher = sharper): {result['blur_score']:.1f}")
                    st.write(f"- Glare in search region: {result['glare_pct']:.1f}%")
                    st.write(f"- Region detection method: {result['region_method']}")
                    st.write(f"- Nearest-neighbour RGB distance: {distance:.2f}")

else:
    st.markdown("""
    <div style="text-align:center; padding:2rem; color:#90a4ae;">
      <span style="font-size:3rem;">☝️</span>
      <p>Select a test type above to get started.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#b0bec5; font-size:.78rem;">
  Metal Concentration Analyzer &nbsp;|&nbsp; No API needed — pure Python RGB extraction
</p>
""", unsafe_allow_html=True)
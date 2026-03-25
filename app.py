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
    .result-box h2 { color: #2e7d32; font-size: 2.4rem; margin: 0; }
    .result-box p  { color: #388e3c; margin: .4rem 0 0; font-size: 1rem; }
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
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧪 Metal Concentration Analyzer</h1>
  <p>Upload a test-tube image → RGB extracted from centre of solution → matched to concentration</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "test_type" not in st.session_state:
    st.session_state.test_type = None

# ── Load CSVs ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not load {path}: {e}")
        return None

al_df = load_csv("al_data.csv")
mn_df = load_csv("mn_data.csv")

# ── RGB extraction (pure PIL) ──────────────────────────────────────────────────
def extract_rgb(img: Image.Image, crop_pct: float = 0.20) -> tuple[int, int, int]:
    """
    Crops the centre (crop_pct x crop_pct) of the image — the liquid zone —
    converts to RGB and returns the median R, G, B across all pixels in that crop.
    Median is used instead of mean to ignore bright glare/dark edge pixels.
    """
    img_rgb = img.convert("RGB")
    w, h    = img_rgb.size

    # Centre crop: middle 20% width, middle 40% height (tall test-tube shape)
    x0 = int(w * (0.5 - crop_pct))
    x1 = int(w * (0.5 + crop_pct))
    y0 = int(h * 0.30)
    y1 = int(h * 0.70)

    cropped = img_rgb.crop((x0, y0, x1, y1))
    pixels  = np.array(cropped).reshape(-1, 3)

    # Remove near-white (glare) and near-black (shadows) pixels
    brightness = pixels.mean(axis=1)
    mask = (brightness > 30) & (brightness < 240)
    filtered = pixels[mask] if mask.sum() > 10 else pixels

    r = int(np.median(filtered[:, 0]))
    g = int(np.median(filtered[:, 1]))
    b = int(np.median(filtered[:, 2]))
    return r, g, b, (x0, y0, x1, y1)

# ── Closest match ──────────────────────────────────────────────────────────────
def find_closest(df, r, g, b):
    diffs = np.sqrt(
        (df["R"] - r)**2 +
        (df["G"] - g)**2 +
        (df["B"] - b)**2
    )
    idx = diffs.idxmin()
    return df.loc[idx], float(diffs[idx])

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
    unit     = "µM"

    st.markdown("---")
    st.markdown(f'<span class="step-label">STEP 2 — Upload Test Tube Image ({element})</span>',
                unsafe_allow_html=True)
    st.info(f"Selected: **{element} Test** — make sure the test tube is centred in the photo.")

    uploaded_img = st.file_uploader(
        "Upload test tube photo (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="tube_image"
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
            st.markdown('<div class="crop-box">📐 RGB will be sampled from the <strong>centre 20% width / middle 40% height</strong> of the image — the liquid zone.</div>',
                        unsafe_allow_html=True)

        # ── Step 3 — Analyse ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<span class="step-label">STEP 3 — Analyse</span>', unsafe_allow_html=True)

        if df is None:
            st.error("❌ Could not load concentration data. Check your CSV files.")
        else:
            if st.button("🔬 Analyse Concentration", type="primary", width="stretch"):
                with st.spinner("Extracting colour from image…"):
                    r, g, b, crop_coords = extract_rgb(img)

                # Show the crop region
                img_crop = img.convert("RGB").crop(crop_coords)
                hex_col  = f"#{r:02x}{g:02x}{b:02x}"
                lum      = 0.299*r + 0.587*g + 0.114*b
                txt_col  = "#000" if lum > 128 else "#fff"

                ca, cb = st.columns([1, 2])
                with ca:
                    st.image(img_crop, caption="Sampled region", width="stretch")
                with cb:
                    st.markdown(f"""
                    <div style="text-align:center; padding:1rem;">
                      <p style="margin:0; color:#555; font-size:.9rem;">🎨 Extracted solution colour</p>
                      <span class="rgb-swatch" style="background:{hex_col}; color:{txt_col};">
                        RGB ({r}, {g}, {b}) &nbsp;|&nbsp; {hex_col}
                      </span>
                      <p style="color:#888; font-size:.8rem; margin:.4rem 0 0;">
                        Median of centre pixels (glare & shadows removed)
                      </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Find closest match
                best_row, distance = find_closest(df, r, g, b)
                concentration  = best_row[conc_col]
                matched_rgb    = (int(best_row["R"]), int(best_row["G"]), int(best_row["B"]))
                matched_hex    = "#{:02x}{:02x}{:02x}".format(*matched_rgb)
                matched_lum    = 0.299*matched_rgb[0] + 0.587*matched_rgb[1] + 0.114*matched_rgb[2]
                matched_txt    = "#000" if matched_lum > 128 else "#fff"

                # Format concentration label
                if concentration == 0:
                    conc_label = "Blank (0 µM)"
                else:
                    conc_label = f"{int(concentration)} µM"

                # ── Confidence calculation ─────────────────────────────────
                MAX_DIST = 80.0  # distances beyond this = 0% confidence
                confidence_pct = max(0, int((1 - distance / MAX_DIST) * 100))

                if distance < 15:
                    conf_label  = "High"
                    conf_icon   = "✅"
                    conf_color  = "#2e7d32"
                    bar_color   = "#43a047"
                    conf_msg    = None
                elif distance < 35:
                    conf_label  = "Medium"
                    conf_icon   = "⚠️"
                    conf_color  = "#e65100"
                    bar_color   = "#fb8c00"
                    conf_msg    = "The colour match is approximate. Try retaking the photo with more even lighting."
                else:
                    conf_label  = "Low"
                    conf_icon   = "❌"
                    conf_color  = "#c62828"
                    bar_color   = "#e53935"
                    conf_msg    = None  # shown separately as warning box

                # Result card — only show if confidence not critically low
                if distance < MAX_DIST:
                    st.markdown(f"""
                    <div class="result-box" style="{'opacity:0.6;' if distance >= 35 else ''}">
                      <p style="color:#1b5e20; font-size:.9rem; margin-bottom:.3rem;">🔍 Closest match found</p>
                      <h2>{conc_label}</h2>
                      <p>{element} concentration</p>
                      <hr style="border:1px solid #a5d6a7; margin:1rem 0;">
                      <p style="margin:0; font-size:.85rem; color:#388e3c;">
                        Matched reference colour:&nbsp;
                        <span class="rgb-swatch" style="background:{matched_hex}; color:{matched_txt}; font-size:.8rem; padding:.2rem .7rem;">
                          RGB ({matched_rgb[0]}, {matched_rgb[1]}, {matched_rgb[2]})
                        </span>
                        &nbsp;| RGB distance: <strong>{distance:.1f}</strong>
                      </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:#ffebee; border:2px solid #e53935; border-radius:14px;
                                padding:1.5rem; text-align:center; margin-top:1.5rem;">
                      <span style="font-size:2.5rem;">🚫</span>
                      <h3 style="color:#c62828; margin:.5rem 0 .3rem;">No reliable match found</h3>
                      <p style="color:#b71c1c; margin:0;">RGB distance is {distance:.1f} — too far from any reference colour.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Confidence meter ───────────────────────────────────────────
                st.markdown(f"""
                <div style="background:white; border:2px solid #e3eaf5; border-radius:14px;
                            padding:1.4rem 1.6rem; margin-top:1.2rem;">
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:.6rem;">
                    <span style="font-weight:700; font-size:1rem; color:#1a237e;">
                      {conf_icon}&nbsp; Prediction Confidence
                    </span>
                    <span style="font-weight:800; font-size:1.3rem; color:{conf_color};">
                      {confidence_pct}% &nbsp;<span style="font-size:.9rem; font-weight:600;">{conf_label}</span>
                    </span>
                  </div>

                  <!-- track -->
                  <div style="background:#e0e0e0; border-radius:99px; height:18px; width:100%; overflow:hidden;">
                    <!-- fill -->
                    <div style="background:linear-gradient(90deg, {bar_color}, {bar_color}cc);
                                width:{confidence_pct}%; height:100%; border-radius:99px;
                                transition:width .4s ease;"></div>
                  </div>

                  <!-- scale labels -->
                  <div style="display:flex; justify-content:space-between;
                              font-size:.72rem; color:#90a4ae; margin-top:.35rem;">
                    <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
                  </div>

                  <!-- threshold markers legend -->
                  <div style="display:flex; gap:1.2rem; margin-top:.9rem; flex-wrap:wrap;">
                    <span style="font-size:.78rem; color:#2e7d32;">
                      🟢 High ≥ 81%&nbsp;(distance &lt; 15)
                    </span>
                    <span style="font-size:.78rem; color:#e65100;">
                      🟠 Medium 56–80%&nbsp;(distance 15–35)
                    </span>
                    <span style="font-size:.78rem; color:#c62828;">
                      🔴 Low &lt; 56%&nbsp;(distance &gt; 35)
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Warning / tip box based on confidence ──────────────────────
                if distance >= MAX_DIST:
                    st.error(
                        "❌ **Image quality too low for a reliable result.**\n\n"
                        "The extracted colour does not match any reference in the dataset. "
                        "Please retake the photo and ensure:\n"
                        "- The test tube is **centred** and fills most of the frame\n"
                        "- Lighting is **uniform** — avoid shadows or direct flash\n"
                        "- The solution is **fully mixed** before photographing\n"
                        "- Use a **plain white or light background**"
                    )
                elif distance >= 35:
                    st.warning(
                        "⚠️ **Low confidence — consider retaking the photo.**\n\n"
                        f"{conf_msg}\n\n"
                        "Tips for a better result:\n"
                        "- Shoot in natural daylight or under a consistent lab light\n"
                        "- Keep the test tube **vertical and centred**\n"
                        "- Avoid reflections on the glass"
                    )
                elif distance >= 15:
                    st.info(f"ℹ️ {conf_msg}")

                with st.expander("📋 View full reference data"):
                    st.dataframe(df, width="stretch")

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
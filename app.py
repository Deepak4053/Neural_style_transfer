import streamlit as st
from PIL import Image
from io import BytesIO
from model import run_style_transfer

st.set_page_config(page_title="Neural Style Transfer", layout="wide")

#  Header 
st.title("🎨 Neural Style Transfer")
st.markdown("Upload a **content image** and a **style image** to blend them into a masterpiece.")
st.divider()

#  Upload Section 
col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 Content Image")
    content_file = st.file_uploader("The subject of your artwork", type=["jpg", "jpeg", "png"])
    if content_file:
        content_img = Image.open(content_file).convert("RGB").resize((400, 400))
        st.image(content_img, caption="Content (400×400)", width=400)

with col2:
    st.subheader("🖼️ Style Image")
    style_file = st.file_uploader("The painting or texture to apply", type=["jpg", "jpeg", "png"])
    if style_file:
        style_img = Image.open(style_file).convert("RGB").resize((400, 400))
        st.image(style_img, caption="Style (400×400)", width=400)

st.divider()

#  Controls 
st.subheader("⚙️ Settings")
steps = st.slider(
    "Training Steps — more steps = stronger style blending",
    min_value=100, max_value=500, value=200, step=50
)
st.caption(f"ℹ️ {steps} steps selected. Higher values take longer but produce richer results.")

st.divider()

#  Run 
if content_file and style_file:
    if st.button("✨ Generate Masterpiece", type="primary", use_container_width=True):

        progress_bar = st.progress(0, text="Starting optimization...")
        status_text  = st.empty()

        def progress_cb(p):
            step_num = int(p * steps)
            progress_bar.progress(p, text=f"Optimizing: step {step_num}/{steps}")
            status_text.caption(f"⏳ {int(p * 100)}% complete")

        try:
            result = run_style_transfer(
                content_img, style_img,
                steps=steps,
                progress_cb=progress_cb
            )

            progress_bar.progress(1.0, text="Done!")
            status_text.empty()

            st.success("✅ Style transfer complete!")
            st.divider()

            #  Result 
            st.subheader("🖼️ Result")

            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                st.image(content_img, caption="Original Content",   width=400)
            with r_col2:
                st.image(style_img,   caption="Style Reference",    width=400)
            with r_col3:
                st.image(result,      caption="✨ Stylized Output", width=400)

            #  Download 
            st.divider()
            buf = BytesIO()
            result.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Result (PNG)",
                data=buf.getvalue(),
                file_name="masterpiece.png",
                mime="image/png",
                use_container_width=True,
                type="secondary"
            )

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Something went wrong: {e}")

elif not content_file and not style_file:
    st.info("👆 Upload both images above to get started.")
elif not content_file:
    st.warning("⚠️ Please upload a content image.")
else:
    st.warning("⚠️ Please upload a style image.")
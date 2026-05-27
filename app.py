import streamlit as st
import os
import cv2
import tempfile
from PIL import Image
from inference.predict import DeepfakePredictor

st.set_page_config(page_title="Deepfake Detection Explorer", layout="wide")

st.title("Deepfake Detection System")
st.write("Upload an image or video to check if it's a deepfake and view the Explainable AI (XAI) analysis.")

@st.cache_resource
def load_predictor():
    checkpoint_path = "checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        return None
    return DeepfakePredictor(checkpoint_path=checkpoint_path, device="cpu") # Use CPU for simple web explorer

predictor = load_predictor()

if predictor is None:
    st.warning("⚠️ Model checkpoint not found at `checkpoints/best_model.pth`. Please train the model or place a checkpoint there to enable predictions.")
else:
    uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.write("---")
        st.write("### Analysis Results")
        
        with st.spinner("Analyzing..."):
            try:
                if file_ext in [".mp4", ".avi", ".mov"]:
                    st.video(tmp_path)
                    result = predictor.predict_video(tmp_path, compute_xai=True)
                else:
                    st.image(tmp_path, caption="Uploaded Image", use_column_width=True)
                    result = predictor.predict_image(tmp_path, compute_xai=True)

                st.write(f"**Prediction:** {result['label']}")
                st.write(f"**Probability:** {result['prediction']:.4f}")
                st.write(f"**Confidence:** {result['confidence']}")
                
                if 'visualization' in result and result['visualization'] is not None:
                    st.write("### XAI Visualization")
                    st.write("This visualization shows which parts of the image/video influenced the model's decision.")
                    # OpenCV uses BGR, convert to RGB for Streamlit
                    vis_rgb = cv2.cvtColor(result['visualization'], cv2.COLOR_BGR2RGB)
                    st.image(vis_rgb, use_column_width=True)
            except Exception as e:
                st.error(f"Error during analysis: {e}")
            finally:
                os.unlink(tmp_path)

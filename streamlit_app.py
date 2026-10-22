import streamlit as st
from src.predict import ToxicPredictor

st.set_page_config(page_title="Toxic Comment Detection", page_icon="🛡️", layout="centered")
st.title("Explainable Toxic Comment Detection")
st.write("Enter a comment to estimate scores across six toxicity categories.")
st.caption("Research prototype only. Scores are model estimates, not definitive judgments.")
@st.cache_resource
def get_predictor():
    return ToxicPredictor()
text = st.text_area("Comment", height=150, placeholder="Type or paste a comment...")
if st.button("Analyze comment", type="primary"):
    if not text.strip():
        st.warning("Please enter a comment.")
    else:
        try:
            results = get_predictor().predict(text)
            for result in results:
                st.write(f"**{result['label']}** — {result['score']:.3f}")
                st.progress(min(1.0, max(0.0, result["score"])))
            flagged = [r["label"] for r in results if r["flagged"]]
            st.subheader("Threshold-based result")
            st.write(", ".join(flagged) if flagged else "No category crossed the default 0.5 threshold.")
        except Exception as exc:
            st.error(str(exc))

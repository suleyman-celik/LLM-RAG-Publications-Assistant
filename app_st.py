import os
import uuid
import requests
import logging

# https://docs.streamlit.io/get-started/tutorials/create-an-app
import streamlit as st

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------- Config ----------------
BASE_URL = f"http://{('app' if os.path.exists('/.dockerenv') else 'localhost')}:5000"
logger.info(f"Backend API: {BASE_URL}")

st.set_page_config(page_title="WHO Publications Assist API", page_icon="🤖", layout="wide")
st.title("🤖 [WHO Publications](https://www.who.int/europe/publications/i) Assist")

# ---------------- Session State Initialize/Store----------------
# For question, answer, and conversation ID
# if "question" not in st.session_state:
#     st.session_state.question = ""
defaults = {
    "question": "",
    "answer": "",
    "conversation_id": "",
    "model": "phi3:latest",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- API Helpers ----------------
# Function to ask a question to the API
def ask_question(url, question, model="phi3:latest"):
    """Send question to API and return JSON response."""
    try:
        resp = requests.post(
            f"{url}/question",
            # json={"question": question, "model": model},
            json={"question": question},
            # timeout=120
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"answer": "⚠️ API request timed out."}
    except requests.exceptions.RequestException as e:
        logger.exception("Error calling API")
        return {"answer": f"❌ API error: {e}"}


# ---------------- UI ----------------
# Main: Question input + Answer
st.session_state.question = st.text_input(
    "Enter your question:",
    value=st.session_state.question or "adult population in Ukraine",

)
if st.button("Get Answer", type="primary"):
    if st.session_state.question.strip():
        # with st.spinner(f"Querying {st.session_state.model}..."):
        with st.spinner("Querying API..."):
            response = ask_question(BASE_URL, st.session_state.question, st.session_state.model)
        st.session_state.answer = response.get("answer", "No answer provided")
        st.session_state.conversation_id = response.get("conversation_id", str(uuid.uuid4()))
    else:
        st.warning("❗ Please enter a question.")


# Main: Show answer
if st.session_state.answer:
    st.subheader("Answer:")
    st.write(st.session_state.answer)
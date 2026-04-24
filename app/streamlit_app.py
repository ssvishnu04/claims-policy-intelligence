import sys
from pathlib import Path
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="Claims & Policy Intelligence Platform",
    page_icon="",
    layout="wide"
)

st.title("Claims & Policy Intelligence Platform")
st.caption("P&C Insurance RAG Assistant using LangChain, FAISS, Hugging Face Embeddings, and Groq")

st.markdown("""
This app helps claims adjusters and underwriters retrieve coverage details, exclusions,
claim context, and supporting evidence from insurance documents.
""")

with st.sidebar:
    st.header("Claim Context")
    policy_id = st.text_input("Policy ID", value="POL-1001")
    claim_id = st.text_input("Claim ID", value="CLM-2001")

    st.markdown("---")
    st.markdown("### Sample Questions")
    st.markdown("""
    - Is this claim covered?
    - What exclusions apply?
    - What deductible should be reviewed?
    - What supporting documents are relevant?
    - What should the adjuster verify next?
    """)

question = st.text_area(
    "Ask a claims or policy question",
    value="Is this sudden pipe burst water damage covered and what exclusions apply?",
    height=120
)

if st.button("Analyze Claim", type="primary"):
    if not policy_id.strip() or not claim_id.strip() or not question.strip():
        st.error("Please enter Policy ID, Claim ID, and a question.")
    else:
        try:
            from app.rag_pipeline import ask_claims_assistant

            with st.spinner("Retrieving relevant insurance documents and generating answer..."):
                result = ask_claims_assistant(
                    question=question,
                    policy_id=policy_id,
                    claim_id=claim_id
                )

            st.subheader("AI Answer")
            st.write(result["answer"])

            st.subheader("Retrieved Sources")

            for i, source in enumerate(result["sources"], start=1):
                metadata = source.metadata
                with st.expander(
                    f"Source {i}: {metadata.get('filename')} | {metadata.get('document_type')}"
                ):
                    st.write("**Metadata:**")
                    st.json(metadata)

                    st.write("**Content:**")
                    st.write(source.page_content)

        except Exception as e:
            st.error("The RAG app failed while processing.")
            st.exception(e)
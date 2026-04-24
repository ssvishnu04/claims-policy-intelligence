import sys
from pathlib import Path
import pandas as pd
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

    policy_id = st.text_input(
        "Policy ID",
        placeholder="Example: POL-1001"
    )

    claim_id = st.text_input(
        "Claim ID",
        placeholder="Example: CLM-2001"
    )

    st.markdown("---")
    st.markdown("### Sample Questions")
    st.markdown("""
    - Is this claim covered?
    - What exclusions apply?
    - What deductible should be reviewed?
    - Give me complete details of this claim in a table
    - What should the adjuster verify next?
    """)

question = st.text_area(
    "Ask a claims or policy question",
    placeholder="Example: Give me complete details of claim CLM-2001 in a table structure.",
    height=120
)

if st.button("Analyze Claim", type="primary"):
    if not policy_id.strip() or not claim_id.strip() or not question.strip():
        st.error("Please enter Policy ID, Claim ID, and a question.")
    else:
        try:
            from app.rag_pipeline import ask_claims_assistant

            with st.spinner("Retrieving structured claim data and policy context..."):
                result = ask_claims_assistant(
                    question=question,
                    policy_id=policy_id,
                    claim_id=claim_id,
                )

            st.subheader("AI Answer")
            st.write(result["answer"])

            if "claim_profile" in result:
                profile = result["claim_profile"]

                st.subheader("Structured Claim Profile")

                claim_summary = {
                    "Policy ID": profile.get("policy_id"),
                    "Claim ID": profile.get("claim_id"),
                    "Loss Date": profile.get("claims_history", {}).get("loss_date"),
                    "Loss Type": profile.get("claims_history", {}).get("loss_type"),
                    "Claim Status": profile.get("claims_history", {}).get("claim_status"),
                    "Paid Amount": profile.get("claims_history", {}).get("paid_amount"),
                    "Repair Estimate Total": profile.get("repair_estimate_total"),
                }

                st.table(pd.DataFrame([claim_summary]))

                fnol = profile.get("fnol", {})
                if fnol:
                    with st.expander("FNOL Details"):
                        st.json(fnol)

                repair_estimates = profile.get("repair_estimates", [])
                if repair_estimates:
                    st.write("**Repair Estimates**")
                    st.dataframe(pd.DataFrame(repair_estimates), use_container_width=True)

            st.subheader("Retrieved Sources")

            if not result["sources"]:
                st.warning("No matching source documents found for this Policy ID and Claim ID.")
            else:
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
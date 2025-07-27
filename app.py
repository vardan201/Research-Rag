import streamlit as st
from rag_pipeline import final_chain  # Import your final RAG chain

st.set_page_config(page_title="🧠 RAG Q&A", layout="centered")
st.title("📚 Ask your AI Research Assistant")

# Input box
query = st.text_input("Ask a question based on the PDF/papers:", placeholder="e.g. What is BERT?")

# When user submits a query
if query:
    with st.spinner("Searching and thinking..."):
        try:
            # Send the query directly as a string
            response = final_chain.invoke(query)

            # Handle response types (string or dict)
            if isinstance(response, str):
                answer = response.strip()
            elif isinstance(response, dict):
                # Optional: Try known keys
                answer = response.get("answer") or response.get("result") or str(response)
                answer = answer.strip()
            else:
                answer = str(response).strip()

            st.success("Answer:")
            st.write(answer)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

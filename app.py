import streamlit as st
from rag_pipeline import build_rag_chain

# Initialize Streamlit session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = build_rag_chain()

# Streamlit app layout
st.title("Enterprise Knowledge Assistant")
st.markdown("Ask questions about your enterprise documents and get concise answers.")

# Input field for user query
user_query = st.text_input("Enter your question:", placeholder="What is the refund policy?")

# Button to submit the query
if st.button("Submit"):
    if user_query.strip():
        with st.spinner("Processing your query..."):
            try:
                # Query the RAG chain
                result = st.session_state.rag_chain({"query": user_query})
                answer = result["result"]
                sources = result["source_documents"]

                # Display the answer
                st.success("Answer:")
                st.write(answer)

                # Display the sources
                st.markdown("### Sources:")
                for doc in sources:
                    st.write(f"- {doc.metadata['source']}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid question.")
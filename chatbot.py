import streamlit as st
from streamlit_pills import pills
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, ServiceContext
from pathlib import Path
import os

os.environ['OPENAI_API_KEY'] = "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"


# Create a llama session
service_context = ServiceContext.from_defaults(chunk_size_limit=256)

# Load your data to prepare data for creating embeddings
documents = SimpleDirectoryReader(input_files=['sher.txt']).load_data()

# Embeddings are created here for your data
global_index = GPTSimpleVectorIndex.from_documents(documents)

# You can save your indexes, to be re-used again
global_index.save_to_disk('sherlockholmes.json')

# To load vectors from local/ Allows to extract it from server also
global_index = GPTSimpleVectorIndex.load_from_disk(f'sherlockholmes.json', service_context=service_context)

st.subheader("AI Assistant based on Custom Knowledge Base: `The Adventure of Sherlock Holmes`")

# You can also use radio buttons instead
selected = pills("", ["OpenAI", "Huggingface"], ["🤖", "🤗"])

user_input = st.text_input("You: ",placeholder = "Ask me anything ...", key="input")

if st.button("Submit", type="primary"):
    st.markdown("----")
    res_box = st.empty()

    if selected == "OpenAI":
        response = global_index.query(user_input, similarity_top_k=3)
        res_box.write(str(response))
        print(response.source_nodes)

    else:
        res_box.write("Work in progress!!")

st.markdown("----")

import os
import argparse
import warnings
import streamlit as st

from model import get_llm
from read_docs import read_docs

warnings.filterwarnings('ignore')

# args
def args_parser():
    parser = argparse.ArgumentParser("RAG robot", add_help = False)
    
    # basic
    parser.add_argument("--filepath", default = "./data", type = str)

    # model settings
    parser.add_argument("--llm", default = "meta-llama/Llama-2-7b-chat-hf", type = str)
    parser.add_argument("--embed_model", default = "BAAI/bge-small-en-v1.5", type = str)
    parser.add_argument("--context_window", default = 4096, type = int)
    parser.add_argument("--max_new_tokens", default = 2048, type = int)
    parser.add_argument("--temperature", default = 0.0, type = float)
    parser.add_argument("--chunk_size", default = 1024, type = int)
    parser.add_argument("--memory_token_limit", default = 1500, type = int)

    parser = parser.parse_args()

    return parser

args = args_parser()

def main():
    # Initialize the app and the chat message history
    st.header("Chat with docs!")
    if "messages" not in st.session_state.keys(): 
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about the documents!"}
        ]

    # uploaded_files
    uploaded_files = st.file_uploader("Upload files here!", accept_multiple_files = True)
    if uploaded_files is not None:
        if not os.path.exists(args.filepath):
            os.mkdir(args.filepath)
        for uploaded_file in uploaded_files:
            with open(f"{args.filepath}/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())

    Settings = get_llm(args)
    index, memory = read_docs(args, Settings)

    chat_engine = index.as_chat_engine(
        chat_mode = "context", 
        memory = memory, 
        system_prompt = (
            "You are a chatbot, able to have normal interactions, as well as talk"
            " about documents."
            ), 
        verbose = True)

    # Input prompt and save to chat history
    if prompt := st.chat_input("Your question"): 
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the chat message history
    for message in st.session_state.messages: 
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Add response to message history
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) 


if __name__ == '__main__':
    main()
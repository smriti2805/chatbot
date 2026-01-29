import streamlit as st
import os
from huggingface_hub import InferenceClient
from chatbot import scrape_website

# Streamlit Page Config
st.set_page_config(page_title="URL Chatbot", layout="wide")

st.title("🌐 URL Chatbot")
st.markdown("Chat with any website using Hugging Face models.")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "website_content" not in st.session_state:
    st.session_state.website_content = None
if "hf_client" not in st.session_state:
    st.session_state.hf_client = None

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key Input
    api_key = st.text_input("Hugging Face API Key", type="password")
    
    # URL Input
    url = st.text_input("Website URL")
    
    if st.button("Load Website"):
        if not api_key:
            st.error("Please enter a Hugging Face API Key.")
        elif not url:
            st.error("Please enter a URL.")
        else:
            with st.spinner("Scraping website..."):
                content = scrape_website(url)
                if content:
                    st.session_state.website_content = content
                    try:
                        st.session_state.hf_client = InferenceClient(
                            model="meta-llama/Meta-Llama-3-8B-Instruct",
                            token=api_key
                        )
                        st.success("Website loaded successfully!")
                        # Clear previous chat history when new site matches
                        # st.session_state.messages = [] 
                        # Optional: keep history or clear it? Clearing makes sense for new context.
                        st.session_state.messages = []
                    except Exception as e:
                        st.error(f"Failed to initialize client: {e}")
                else:
                    st.error("Failed to scrape website content.")

# Chat Interface
if st.session_state.website_content:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask a question about the website..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Construct conversation for the model
            # System prompt with context
            messages_payload = [
                {"role": "system", "content": f"You are a helpful assistant. Use the following website content to answer questions. Content: {st.session_state.website_content[:15000]}"} # Limit context if needed, though scrape limits to 10k
            ]
            # Add history (limit to last few turns to save tokens if needed, but 8B handle ~8k context often)
            # The simple chatbot.py used full history. We'll pass full history here.
            for msg in st.session_state.messages:
                messages_payload.append(msg)

            try:
                # Stream response
                stream = st.session_state.hf_client.chat.completions.create(
                    messages=messages_payload,
                    max_tokens=500,
                    stream=True
                )
                
                for chunk in stream:
                    # Check if choices exist and are not empty
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
else:
    st.info("Please enter an API Key and URL in the sidebar to start chatting.")

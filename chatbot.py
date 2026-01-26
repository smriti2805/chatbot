"""
pip install requests
pip install beautifulsoup4
pip install huggingface-hub

"""

import os
import sys
import requests
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient

# --- 1. Environment & Setup Helpers ---

def get_api_key():
    """Retrieves the API key from environment or user input."""
    hf_api_key = os.environ.get("HF_API_KEY")

    if not hf_api_key:
        print("HF_API_KEY not found in environment variables.")
        hf_api_key = input("Please enter your Hugging Face API Key: ").strip()
    return hf_api_key

# --- 2. Extracting Data ---

def scrape_website(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        text = soup.get_text(separator=' ')
        
        # --- 3. Processing Data ---
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text[:10000] 

    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during scraping: {e}")
        return None

# --- 4. Implementing the Chatbot Logic ---

def chat_with_site(client, context_text):
    """
    Runs the chat loop using the provided context.
    """
    print("\n" + "="*50)
    print("Chatbot Initialized! You can now ask questions about the website.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    conversation_history = [
        {"role": "system", "content": f"You are a helpful assistant. You have been provided with the following content from a website. Use this content to answer the user's questions. If the answer is not in the content, say so.\n\nWebsite Content:\n{context_text}"}
    ]

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue

            conversation_history.append({"role": "user", "content": user_input})

            completion = client.chat.completions.create(
                messages=conversation_history,
                max_tokens=200
            )

            assistant_response = completion.choices[0].message.content
            print(f"Bot: {assistant_response}")

            conversation_history.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

# --- 5. Console Demonstration (Main Entry Point) ---

if __name__ == "__main__":
    print("Welcome to the URL Chatbot.")
    
    # Setup
    key = get_api_key()
    if not key:
        print("API Key is required to proceed. Exiting.")
        sys.exit(1)
        

        
    client = InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        token=key
    )


    # Input URL
    url = input("Please enter the website URL to interact with: ").strip()
    
    print("Fetching and processing website content...")
    website_content = scrape_website(url)

    if website_content:
        print(f"Successfully scraped {len(website_content)} characters of content.")
        chat_with_site(client, website_content)
    else:
        print("Failed to retrieve content. Exiting.")

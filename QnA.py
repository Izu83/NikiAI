import streamlit as st
import ollama

TEMPERATURE = 1

ollama_api_key = st.secrets[r"C:\Users\niki\.ollama\id_ed25519.pub"]
print(r"C:\Users\niki\.ollama\id_ed25519.pub")


def chat_with_mistral(prompt, temperature):
    try:
        response = ollama.chat(
            model='mistral',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': temperature}
        )
        return response.get('message', {}).get('content', 'No valid response')
    except Exception as e:
        return f"An error occurred: {e}"

st.title("Chat with Niki AI")

user_input = st.text_area("Enter your prompt:")

if st.button("Send"):
    if user_input.strip():
        response = chat_with_mistral(user_input, TEMPERATURE)
        st.subheader("Niki AI Response:")
        st.write(response)
    else:
        st.warning("Please enter a prompt before sending.")



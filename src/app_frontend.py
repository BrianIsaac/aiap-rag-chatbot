import gradio as gr
import requests
from sseclient import SSEClient
import json
import os

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/query-stream")

def stream_query(message, history):
    """
    Sends a question to the FastAPI /query-stream endpoint and yields streaming chunks.
    """
    url = API_URL
    headers = {"Accept": "text/event-stream"}
    payload = {"question": message}

    # Streaming POST request
    response = requests.post(url, json=payload, headers=headers, stream=True)

    # SSE client parses streamed response
    client = SSEClient(response)

    full_output = ""

    for event in client.events():
        if event.event == "data":
            try:
                # Fix: decode JSON-encoded strings to avoid quotes and \u escapes
                token_piece = json.loads(event.data)
                full_output += token_piece
                yield full_output
            except json.JSONDecodeError:
                continue
        elif event.event == "end":
            break

# Gradio chat interface with type='messages' to avoid deprecation
chat_interface = gr.ChatInterface(
    fn=stream_query,
    type="messages",  # Required to avoid warnings and use OpenAI-style format
    title="📚 AIAP RAG Chatbot",
    chatbot=gr.Chatbot(label="Tutor Bot"),
    textbox=gr.Textbox(placeholder="Ask a question...", label="Your query"),
    examples=[
        "What is AIAP?",
        "How can I prepare for application?",
        "What can I expect in terms of career support?"
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    chat_interface.launch(server_name="0.0.0.0", server_port=7860)

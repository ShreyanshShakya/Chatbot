from flask import Flask, request, jsonify, render_template_string
import google.generativeai as genai
import os
from dotenv import load_dotenv
import markdown # For rendering markdown in HTML

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

MODEL_NAME = "gemini-1.5-flash-latest" # Or your chosen model
safety_settings = [ # Same settings as before
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
model = genai.GenerativeModel(model_name=MODEL_NAME, safety_settings=safety_settings)

app = Flask(__name__)

# Store conversations in memory (simple approach, not persistent across restarts/scales)
# For production, use a database or persistent storage
conversations = {}

# Basic HTML template for the chat interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Gemini Chatbot</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        #chatbox { height: 400px; border: 1px solid #ccc; overflow-y: scroll; padding: 10px; margin-bottom: 10px; }
        .message { margin-bottom: 10px; }
        .user { text-align: right; color: blue; }
        .bot { text-align: left; color: green; }
        #userInput { width: 80%; padding: 10px; }
        #sendButton { padding: 10px; }
    </style>
</head>
<body>
    <h1>Gemini Chatbot</h1>
    <div id="chatbox">
        <!-- Chat messages will appear here -->
        <div class="message bot">Bot: Hello! How can I help you today?</div>
    </div>
    <input type="text" id="userInput" placeholder="Type your message...">
    <button id="sendButton">Send</button>

    <script>
        const chatbox = document.getElementById('chatbox');
        const userInput = document.getElementById('userInput');
        const sendButton = document.getElementById('sendButton');
        // Use a unique session ID (simple example using random number)
        const sessionId = `session_${Math.random().toString(36).substring(7)}`;

        function addMessage(sender, text, isHtml=false) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', sender.toLowerCase());
            if (isHtml) {
                messageDiv.innerHTML = `<strong>${sender}:</strong> ${text}`; // Render HTML if needed
            } else {
                messageDiv.textContent = `${sender}: ${text}`;
            }
            chatbox.appendChild(messageDiv);
            chatbox.scrollTop = chatbox.scrollHeight; // Scroll to bottom
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            addMessage('You', message);
            userInput.value = ''; // Clear input field
            addMessage('Bot', '<i>Thinking...</i>', true); // Show thinking indicator

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message, session_id: sessionId })
                });

                // Remove the "Thinking..." message
                chatbox.removeChild(chatbox.lastChild);

                if (!response.ok) {
                    const errorData = await response.json();
                    addMessage('Bot', `Error: ${errorData.error || response.statusText}`);
                    return;
                }

                const data = await response.json();
                addMessage('Bot', data.reply, true); // Display bot's reply (allow HTML for markdown)

            } catch (error) {
                 // Remove the "Thinking..." message even on error
                if (chatbox.lastChild && chatbox.lastChild.textContent.includes('Thinking...')) {
                     chatbox.removeChild(chatbox.lastChild);
                }
                console.error('Fetch Error:', error);
                addMessage('Bot', `Network Error: ${error.message}`);
            }
        }

        sendButton.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    session_id = data.get('session_id', 'default_session') # Get session ID

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Get or start chat history for this session
    if session_id not in conversations:
         try:
            # Start a new chat session for this user/session
            conversations[session_id] = model.start_chat(history=[])
            print(f"Started new chat session: {session_id}")
         except Exception as e:
            print(f"Error starting chat session {session_id}: {e}")
            return jsonify({"error": "Could not initialize chat session"}), 500
    else:
        print(f"Continuing chat session: {session_id}")


    chat_session = conversations[session_id]

    try:
        response = chat_session.send_message(user_message)
        # Convert markdown response to HTML
        bot_reply_html = markdown.markdown(response.text)
        return jsonify({"reply": bot_reply_html}) # Return HTML response

    except genai.types.BlockedPromptException:
         return jsonify({"reply": "[Your input was blocked by safety settings.]"}), 200 # Return OK but indicate blocking
    except genai.types.StopCandidateException:
         return jsonify({"reply": "[Response generation stopped unexpectedly.]"}), 200 # Return OK but indicate stopping
    except Exception as e:
        print(f"Error during Gemini API call for session {session_id}: {e}")
        # You might want to reset the chat session here if it becomes corrupted
        # del conversations[session_id]
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # Use environment variable for port, default to 8080 for Cloud Run compatibility
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port) # Set debug=False for production
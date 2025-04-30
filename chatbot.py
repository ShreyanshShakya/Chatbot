import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("ERROR: GOOGLE_API_KEY not found in environment variables.")
    print("Please set the GOOGLE_API_KEY in your .env file.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during configuration: {e}")
    exit()


# --- Model Configuration ---
# See https://ai.google.dev/models/gemini for model names
MODEL_NAME = "gemini-1.5-flash-latest" # Or "gemini-pro" or other compatible models

# Safety settings: Adjust as needed
# Refer to: https://ai.google.dev/docs/safety_setting_guidance
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Generation Configuration (Optional)
generation_config = {
    "temperature": 0.7, # Controls randomness (0=deterministic, 1=more random)
    "top_p": 1.0,       # Nucleus sampling
    "top_k": 1,         # Top-k sampling
    "max_output_tokens": 2048, # Max length of the response
    # "stop_sequences": ["\n\n---"] # Optional sequences to stop generation
}

# --- Initialize the Model ---
try:
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
except Exception as e:
    print(f"Error creating the Generative Model: {e}")
    exit()

# --- Start the Chat ---
# We use start_chat to maintain conversation history
try:
    chat = model.start_chat(history=[
        # Optional: You can prime the chat with initial messages
        # {"role": "user", "parts": ["You are a helpful assistant."]},
        # {"role": "model", "parts": ["Okay, how can I help you today?"]}
    ])
    print("Chatbot initialized. Type 'quit' or 'exit' to end the chat.")
    print("-" * 20)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Chatbot exiting. Goodbye!")
            break

        if not user_input:
            continue

        try:
            # Send the user message and stream the response
            response = chat.send_message(user_input, stream=True)

            print("Bot: ", end="", flush=True)
            for chunk in response:
                # Check if there's text in the chunk (sometimes chunks might be empty or contain metadata)
                if hasattr(chunk, 'text'):
                    print(chunk.text, end="", flush=True)
                # Handle potential errors within the stream (e.g., blocked content)
                if not chunk.parts:
                     print("\n[Blocked by safety settings or other error]", end="")
                     break # Stop printing this response if blocked
            print() # Newline after the full response

            # Optional: Print full candidate details if needed for debugging
            # print("\n--- Response Details ---")
            # print(response.prompt_feedback) # Safety feedback for the prompt
            # print(response.candidates[0].finish_reason) # Why generation stopped
            # print(response.candidates[0].safety_ratings) # Safety rating for the response
            # print("--- End Details ---")


        except genai.types.BlockedPromptException as e:
             print("\n[Your input was blocked by safety settings.]")
             print(e)
        except genai.types.StopCandidateException as e:
             print("\n[Response stopped unexpectedly.]")
             print(e)
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            # Optional: Decide if you want to break the loop on errors
            # break

except Exception as e:
    print(f"Error starting the chat session: {e}")
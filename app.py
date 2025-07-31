# app.py

from flask import Flask, render_template, request, jsonify
from chat import get_final_response # Import your function from chat.py

app = Flask(__name__)

# --- We will keep the conversation state in memory ---
# In a real production app, you might use a database or a more robust session management.
conversation_state = "general"

@app.route("/")
def index():
    """ Renders the main chat page """
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    """ Handles the chat message from the user """
    global conversation_state # Use the global state variable
    
    user_message = request.form["msg"]
    response, new_state = get_final_response(user_message, conversation_state)
    
    # Update the state for the next turn
    conversation_state = new_state
    
    return jsonify({"response": response})

if __name__ == "__main__":
    # NOTE: The model and tokenizer are loaded inside chat.py when it's imported.
    # This ensures they are loaded only once when the application starts.
    print("Starting Flask server...")
    app.run(debug=True)
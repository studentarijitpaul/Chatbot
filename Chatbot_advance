import logging
from transformers import pipeline, Conversation
from datetime import datetime

# ---------------- CONFIG ----------------
LOG_FILE = "chatbot_logs.log"
MODEL_NAME = "microsoft/DialoGPT-medium"
# ----------------------------------------

# Logging setup
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)

class SmartChatBot:
    def __init__(self, model_name=MODEL_NAME):
        try:
            print("🚀 Initializing ChatBot... (this may take a moment)")
            self.chat_pipeline = pipeline("conversational", model=model_name)
            self.memory = Conversation()
            logging.info(f"Chatbot initialized with model: {model_name}")
            print("✅ ChatBot ready! Type 'exit' to quit.\n")
        except Exception as e:
            logging.critical(f"Error initializing model: {e}")
            raise RuntimeError(f"Failed to initialize chatbot: {e}")

    def respond(self, user_input):
        """Generate chatbot response with error handling."""
        try:
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Bye! Have a great day, bro!")
                return None

            self.memory.add_user_input(user_input)
            response = self.chat_pipeline(self.memory)
            bot_reply = response.generated_responses[-1]
            logging.info(f"User: {user_input} | Bot: {bot_reply}")
            return bot_reply

        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "😅 Sorry, I ran into an error while processing that."

def main():
    bot = SmartChatBot()
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            reply = bot.respond(user_input)
            if reply is None:
                break

            print(f"Bot: {reply}\n")

        except KeyboardInterrupt:
            print("\n👋 Exiting ChatBot. Bye bro!")
            break
        except Exception as e:
            logging.critical(f"Fatal error in main loop: {e}")
            print("🚨 Something went wrong, restarting chatbot...\n")

if __name__ == "__main__":
    main()

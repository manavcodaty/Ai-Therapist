import openai
import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()



openai.api_key = os.getenv("CHATGPT")

class ChatBot:
    def __init__(self, engine: str = "gpt-3.5-turbo") -> None:
        self.model = engine
        self.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

    def send_message(self, message: str) -> str:
        assistant_response = ""
        self.conversation_history.append({"role": "user", "content": message})
        # Creates a completion for the provided prompt and parameters.
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.conversation_history,
            )
            # Extract the generated response from the API response.
            assistant_response = response.choices[0].message['content']
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
        except openai.error.AuthenticationError as error:
            print("error: {0}".format(error.args))
        except openai.error.RateLimitError as error:
            print("error: {0}".format(error.args))
        except openai.error.InvalidRequestError as error:
            print("error: {0}".format(error.args))
        except openai.error.PermissionError as error:  # Occurred when you are not allowed to sample a model
            print("error: {0}".format(error.args))
        return assistant_response


def main():
    print("Welcome to ChatBot! Type 'quit' to exit.")
    chatbot = ChatBot()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = chatbot.send_message(user_input)
        print("ChatBot:", response)


if __name__ == "__main__":
    main()
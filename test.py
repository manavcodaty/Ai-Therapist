from transformers import BartTokenizer, BartForConditionalGeneration

# Load pre-trained Bart model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def generate_response(prompt, max_length=150):
    # Tokenize the prompt and generate text
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    print("Welcome to the AI Therapist. How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("AI Therapist: Goodbye! Take care.")
            break
        prompt = f"Therapist: {user_input.strip()} How do you feel about that?"
        response = generate_response(prompt)
        print(f"AI Therapist: {response}")

if __name__ == "__main__":
    main()

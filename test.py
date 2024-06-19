import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2-medium'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

def generate_response(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors='tf')
    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.95,
        top_k=50
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    print("Welcome to the AI Therapist. How can I help you today?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Therapist: It was nice talking to you. Take care!")
            break
        
        prompt = f"The following is a conversation with an AI therapist. The therapist is helpful, empathetic, and patient.\n\nUser: {user_input}\nTherapist:"
        ai_response = generate_response(prompt)
        print(f"Therapist: {ai_response.split('Therapist: ')[-1]}")

if __name__ == "__main__":
    main()
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model_path = "./model_cache"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Test the model
test_prompts = [
    "Hello, how are you?",
    "What is machine learning?",
    "Translate to French: Hello world"
]

print("Testing model responses:")
print("=======================")
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,
        num_return_sequences=1,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}") 
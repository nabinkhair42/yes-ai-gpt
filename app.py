from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import snapshot_download
import torch
import os
import traceback

app = Flask(__name__)

# Define model path
model_name = "NoaiGPT/777"
model_path = "./model_cache"

# Download/Clone the model if it doesn't exist locally
if not os.path.exists(model_path):
    print(f"Downloading model {model_name}...")
    snapshot_download(
        repo_id=model_name,
        local_dir=model_path,
        ignore_patterns=["*.msgpack", "*.h5", "*.safetensors"]
    )
    print("Model downloaded successfully!")

# Load the model and tokenizer from local path
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
print("Model and tokenizer loaded successfully!")

# Add after loading the model
print("\nModel Information:")
print("==================")
print(f"Model Type: {model.config.model_type}")
print(f"Vocabulary Size: {model.config.vocab_size}")
print(f"Hidden Size: {model.config.hidden_size}")
print(f"Number of Layers: {model.config.num_layers if hasattr(model.config, 'num_layers') else model.config.num_hidden_layers}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1000000:.2f}M")

# Print model architecture
print("\nModel Architecture:")
print("==================")
print(model)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if not response:
            return jsonify({'error': 'No response generated'}), 500
            
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())  # This will print the full error traceback
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    try:
        # Try a simple test prompt
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({
            'status': 'success',
            'model_loaded': True,
            'test_response': response
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'model_loaded': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
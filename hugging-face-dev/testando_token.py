import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Download the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Verifica se a GPU está disponível
device = 0 if torch.cuda.is_available() else -1

# Create the pipeline with GPU support
sentimentAnalysis = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

input_text = "I love using Hugging Face models!"
# Predict the sentiment
output = sentimentAnalysis(input_text)

print(f"Sentiment using AutoClasses: {output[0]['label']}")
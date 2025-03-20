from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForSequenceClassification
# Download the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Create the pipeline
sentimentAnalysis = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
input="I love using Hugging Face models!"
# Predict the sentiment
output = sentimentAnalysis(input)

print(f"Sentiment using AutoClasses: {output[0]['label']}")
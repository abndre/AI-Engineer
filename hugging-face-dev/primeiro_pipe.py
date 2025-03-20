from transformers import pipeline

# Criando o pipeline de análise de sentimentos
sentiment_pipeline = pipeline("sentiment-analysis")

# Testando com uma frase
text = "I love using Hugging Face models!"
result = sentiment_pipeline(text)

print(result)

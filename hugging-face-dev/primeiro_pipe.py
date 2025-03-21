from transformers import pipeline

# Criando o pipeline de análise de sentimentos
sentiment_pipeline = pipeline("sentiment-analysis")

# Testando com uma frase
text = "I love using Hugging Face models!"
result = sentiment_pipeline(text)

print(result)

# Create a pipeline
classifier = pipeline(
  task="text-classification", 
  model="abdulmatinomotoso/English_Grammar_Checker"
)

# Predict classification
output = classifier("I will walk dog")

print(output)
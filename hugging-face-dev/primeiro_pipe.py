print("===== parte 1 ======")
from transformers import pipeline

# Criando o pipeline de análise de sentimentos
sentiment_pipeline = pipeline("sentiment-analysis")

# Testando com uma frase
text = "I love using Hugging Face models!"
result = sentiment_pipeline(text)

print(result)

print("===== parte 2 ======")

# Create a pipeline
classifier = pipeline(
  task="text-classification", 
  model="abdulmatinomotoso/English_Grammar_Checker"
)

# Predict classification
output = classifier("I will walk dog")

print(output)

print("===== parte 3 ======")


# Create the pipeline
classifier = pipeline(task="text-classification", model="cross-encoder/qnli-electra-base")

# Predict the output
output = classifier("Where is the capital of France?, Brittany is known for their kouign-amann.")

print(output)

print("===== parte 4 ======")
text='A 75-million-year-old Gorgosaurus fossil is the first tyrannosaur skeleton ever found with a filled stomach.'
# Build the zero-shot classifier
classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")

# Create the list
candidate_labels = ["politics", "science", "sports"]

# Predict the output
output = classifier(text, candidate_labels)

print(f"Top Label: {output['labels'][0]} with score: {output['scores'][0]}")
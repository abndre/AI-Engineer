input_string = 'HOWDY, how aré yoü?'
# Import the AutoTokenizer
from transformers import AutoTokenizer, GPT2Tokenizer, DistilBertTokenizer

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Normalize the input string
output = tokenizer.backend_tokenizer.normalizer.normalize_str(input_string)

print(output)

print("===== parte 2 ======")

input = 'Pineapple on pizza is pretty good, I guess.'
# Download the gpt tokenizer
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenize the input
gpt_tokens = gpt_tokenizer.tokenize(input )

# Repeat for distilbert
distil_tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
distil_tokens = distil_tokenizer.tokenize(text=input)

# Compare the output
print(f"GPT tokenizer: {gpt_tokens}")
print(f"DistilBERT tokenizer: {distil_tokens}")
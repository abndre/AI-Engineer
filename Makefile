ENV_NAME = huggingface_env
PYTHON_VERSION = 3.9

.PHONY: create_env activate install run clean

create_env:
	conda create -n $(ENV_NAME) python=$(PYTHON_VERSION) -y

activate:
	conda activate $(ENV_NAME)

install:
	conda activate $(ENV_NAME) && pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

run:
	conda activate $(ENV_NAME) && python sentiment_analysis.py

clean:
	conda remove --name $(ENV_NAME) --all -y

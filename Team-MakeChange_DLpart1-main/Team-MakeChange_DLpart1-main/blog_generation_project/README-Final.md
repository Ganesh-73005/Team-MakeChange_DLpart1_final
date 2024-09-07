# WikiHow Content Generation

This project provides a machine learning solution for automatically generating headings, summaries, and tags for WikiHow-style articles. It uses state-of-the-art language models to process input paragraphs and produce relevant content.

## Features

- Generate concise headings for input paragraphs
- Create summarized versions of input text
- Produce relevant tags/keywords for the content
- Utilizes fine-tuned LED (Longformer Encoder-Decoder) model for heading and summary generation
- Employs BERT model for tag generation

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- Datasets library
- Pandas
- ROUGE score (for evaluation)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/wikihow-content-generation.git
   cd wikihow-content-generation
   ```

2. Install the required packages:
   ```
   pip install transformers datasets pandas rouge_score torch
   ```

3. Download the fine-tuned models (not included in this repository due to size constraints):
   - Place the fine-tuned LED model in `/content/drive/MyDrive/checkpoint-60/`
   - Place the fine-tuned BERT model in `./fine_tuned_bert_model/`

## Usage

1. Import the necessary modules and load the models:

```python
from transformers import LEDTokenizer, LEDForConditionalGeneration, BertTokenizer, BertForMaskedLM
import torch
import pandas as pd
from datasets import Dataset

tokenizer = LEDTokenizer.from_pretrained("/content/drive/MyDrive/checkpoint-60")
model1 = LEDForConditionalGeneration.from_pretrained("/content/drive/MyDrive/checkpoint-60").to("cuda").half()
model = BertForMaskedLM.from_pretrained('./fine_tuned_bert_model')
```

2. Define the generation functions:

```python
def generate_text(batch):
    # Function implementation (as provided in the code)

def generate_tags(paragraph):
    # Function implementation (as provided in the code)
```

3. Process your input:

```python
sample_paragraph = """Your input paragraph here"""
df_test = pd.DataFrame([sample_paragraph], columns=['Paragraph'])
df_test = Dataset.from_pandas(df_test)
tags = generate_tags(sample_paragraph)
result = df_test.map(generate_text, batched=True, batch_size=2)

print("Generated Heading:", result["generated_heading"])
print("Generated Summary:", result["generated_summary"])
print("Generated Tags:", tags)
```

## Training

The repository includes code for fine-tuning the BERT model on custom data. To train on your own dataset:

1. Prepare your dataset as a text file with one sentence per line.
2. Update the file path in the training code:
   ```python
   dataset = LineByLineTextDataset(
       tokenizer=tokenizer,
       file_path="/path/to/your/dataset.txt",
       block_size=max_length,
   )
   ```
3. Run the training code as provided in the script.

## Output
![image](https://github.com/Ganesh-73005/Team-MakeChange_DLpart1_final/blob/main/Team-MakeChange_DLpart1-main/Team-MakeChange_DLpart1-main/blog_generation_project/Screenshot%202024-09-07%20164646.png)
![image](../1.)


## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

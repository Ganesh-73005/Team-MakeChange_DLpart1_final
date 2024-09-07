!pip install transformers datasets pandas rouge_score

import pandas as pd
import matplotlib.pyplot as plt
from transformers import LEDTokenizer, LEDForConditionalGeneration
import torch
from datasets import load_metric
from transformers import AutoModelForSeq2SeqLM
import numpy as np
from datasets import Dataset, load_metric, load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import transformers

# Load the csv file
file_path = './wikiHow.csv'
df = pd.read_csv(file_path)

# Display first few rows of the dataset
df.head()

# Data preprocessing
# Remove NA values and duplicates
df = df.dropna()
df = df.drop_duplicates()

# Add a new column 'length' to the dataframe, which counts the number of words in each paragraph
df['length'] = df['Paragraph'].map(lambda x: len(x.split(" ")))

# Assign the word count column to a variable
numOfWords = df['length']

# Plot a histogram of the word count distribution
fig = plt.figure(figsize=(5, 3))
plt.hist(numOfWords.to_numpy(), bins=[0, 50, 100, 200, 300, 500, 1000])
plt.title("Word count distribution")
plt.show()

# Remove outliers
tempdf = df[df.length <= 200]
print(tempdf.shape)

# Fine-tuning with LED:

# Import model version
tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")

# Define key parameters
max_input_length = 1024   # Maximum token length for input (paragraphs)
max_output_length = 64    # Maximum token length for output (headings)
batch_size = 16           # Batch size for training

# Function to process data into model inputs
def process_data_to_model_inputs(batch):
    # Tokenize the inputs (paragraphs) and labels (headings)
    inputs = tokenizer(
        batch["Paragraph"],                 # Text of the paragraph
        padding="max_length",               # Pad to max_length
        truncation=True,                    # Truncate if text exceeds max_length
        max_length=max_input_length,        # Set the max input length to 1024
    )

    outputs = tokenizer(
        batch["Subheading"],                   # Text of the heading
        padding="max_length",               # Pad to max_length
        truncation=True,                    # Truncate if heading exceeds max_length
        max_length=max_output_length,       # Set the max output length to 64
    )

    # Assign the tokenized input ids and attention mask to the batch
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # Create global attention mask lists, all set to 0 initially
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # Set the global attention on the first token in the batch to 1
    batch["global_attention_mask"][0][0] = 1

    # Assign the tokenized outputs to the batch
    batch["labels"] = outputs.input_ids

    # Ensure that the PAD token is ignored during the loss computation
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch

# Split the data
import numpy as np
train, validate, test = np.split(tempdf.sample(frac=1, random_state=42), [int(.6*len(df)), int(.7*len(df))])
print(train.shape)
print(validate.shape)
print(test.shape)
validate = validate[:20]
print(validate.shape)

# Create datasets
train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(validate)

# Process datasets for model input
train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["Article Title", "Subheading", "Paragraph", "length", "__index_level_0__"],
)

val_dataset = val_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["Article Title", "Subheading", "Paragraph", "length", "__index_level_0__"],
)

# Set dataset format
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# Load rouge metric
rouge = load_metric("rouge")

# Define compute_metrics function
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid
    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

# Set up training arguments
transformers.logging.set_verbosity_info()

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    output_dir="./",
    logging_steps=5,
    eval_steps=10,
    save_steps=10,
    save_total_limit=2,
    gradient_accumulation_steps=4,
    num_train_epochs=10
)

# Train the model
led = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384", gradient_checkpointing=True, use_cache=False)

led.config.num_beams = 2
led.config.max_length = 64
led.config.min_length = 2
led.config.length_penalty = 2.0
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 3

trainer = Seq2SeqTrainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# Inference
import pandas as pd
from datasets import Dataset, load_metric, load_dataset
from transformers import LEDTokenizer, LEDForConditionalGeneration
import torch

sample_paragraph = "Virat kohli is an inspiration to many people around the world"
data = [sample_paragraph]
df = pd.DataFrame(data, columns=['Paragraph'])
df["Paragraph"][0]
df_test = Dataset.from_pandas(df)
df_test

tokenizer = LEDTokenizer.from_pretrained("/content/checkpoint-60")
model = LEDForConditionalGeneration.from_pretrained("/content/checkpoint-60").to("cuda").half()

def generate_answer(batch):
  inputs_dict = tokenizer(batch["Paragraph"], padding="max_length", max_length=512, return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to("cuda")
  attention_mask = inputs_dict.attention_mask.to("cuda")
  global_attention_mask = torch.zeros_like(attention_mask)
 
  predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
  batch["generated_heading"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  return batch

result = df_test.map(generate_answer, batched=True, batch_size=2)

print(result["generated_heading"])

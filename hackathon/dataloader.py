import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import os
import os
from sklearn.model_selection import train_test_split

# Define your data directories for texts and labels
texts_directory = '/Users/admin/SIHdataset/IN-Abs/train-data/judgement'
labels_directory = '/Users/admin/SIHdataset/IN-Abs/train-data/summary'

# Define a function to load your data
def load_data(texts_dir, labels_dir):
    texts = []
    labels = []

    # Iterate through the text data directory
    for label in os.listdir(texts_dir):
        label_dir = os.path.join(texts_dir, label)
        
        # Check if the subdirectory is a directory
        if os.path.isdir(label_dir):
            # Iterate through text files in the subdirectory
            for filename in os.listdir(label_dir):
                file_path = os.path.join(label_dir, filename)
                
                # Read the text from the file
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                
                texts.append(text)
                labels.append(label)  # You may need to map labels to numeric values

    return texts, labels

# Load the data
texts, labels = load_data(texts_directory, labels_directory)

# Split the data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize and encode the text data using the BERT tokenizer
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Convert the tokenized data to PyTorch tensors
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create train and validation datasets
train_dataset = CustomDataset(train_encodings, train_labels)
validation_dataset = CustomDataset(val_encodings, val_labels)

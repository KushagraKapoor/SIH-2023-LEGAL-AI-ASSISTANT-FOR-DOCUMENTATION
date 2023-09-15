import re
import string
from dataloader import Dataset
# Define a function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text

# Example usage
text = "This is an example sentence with some punctuation!!!"
preprocessed_text = preprocess_text(text)
print(preprocessed_text)

class TextDataset(Dataset):
    def __init__(self, file_paths, tokenizer, max_length=128):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'r', encoding='utf-8') as file:
            text = file.read().strip()

        # Preprocess the text
        text = preprocess_text(text)

        # Tokenize and encode the text
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }




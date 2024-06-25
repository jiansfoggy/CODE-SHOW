import nltk
import numpy as np
import pandas as pd
import random
import string
import torch
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from transformers import RobertaTokenizerFast

code = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'E':5, 'Q':6, 'G':7, 'H':8, 'I':9,
        'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19}

  
#nltk.download('wordnet')
#nltk.download('stopwords')

# task 1: check the token part
class ProteinDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

# Data Augmentation
def normalize_text(text):
    # Define stopwords and stemmer
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text 

# Replace synonyms
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(sentence, num_replacements=1):
    if not isinstance(sentence, str):
        print(f"Unexpected input type: {type(sentence)}, value: {sentence}")
        return sentence  # Return the input as is if it's not a string

    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= num_replacements:
            break

    sentence = ' '.join(new_words)
    return sentence

# for imbalanced data
def get_train_test(file_dir):
    all_file = pd.read_excel(file_dir)
    # Apply the normalization function on the 'text' column of the dataset
    all_file['text'] = all_file['text'].apply(normalize_text)

    # Apply replace sysnonym
    all_file['text'] = all_file['text'].apply(synonym_replacement)

    augmented_texts = []
    augmented_labels = []

    for idx, row in all_file.iterrows():
        augmented_text = synonym_replacement(row['text'])
    
        # Only append if the augmented text is different from the original
        if augmented_text != row['text']:
            augmented_texts.append(augmented_text)
            augmented_labels.append(row['label'])  # Assuming 'label' is the column name for labels

    # Combine the original and augmented data
    all_texts = all_file['text'].tolist() + augmented_texts
    all_label = all_file['label'].tolist() + augmented_labels

    combined_data = pd.DataFrame({
        'text': all_texts,
        'label': all_label  
    })

    # Filter out rows with label value 2
    filtered_data = combined_data[combined_data['label'] != 2]

    train_data, temp_data = train_test_split(filtered_data, test_size=0.2, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Convert labels to integers
    label_encoder = LabelEncoder()
    label_encoder.fit(filtered_data['label'])

    train_labels = label_encoder.transform(train_data['label'])  
    valid_labels = label_encoder.transform(valid_data['label'])
    test_labels  = label_encoder.transform(test_data['label'])

    train_texts  = train_data['text'].tolist()
    valid_texts  = valid_data['text'].tolist()
    test_texts   = test_data['text'].tolist()
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    valid_labels = torch.tensor(valid_labels, dtype=torch.long)
    test_labels  = torch.tensor(test_labels,  dtype=torch.long)
    all_file, train_data_pos, test_data_pos = None, None, None
    train_data_neg, test_data_neg, train_data, test_data = None, None, None, None
    del all_file, train_data_pos, test_data_pos, train_data_neg, test_data_neg, train_data, test_data
    return train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels

def get_dataloader(file_dir, max_len):
    train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels = get_train_test(file_dir)
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    train_dataset = ProteinDataset(train_texts, train_labels, tokenizer, max_len)
    valid_dataset = ProteinDataset(valid_texts, valid_labels, tokenizer, max_len)
    test_dataset  = ProteinDataset(test_texts,  test_labels,  tokenizer, max_len)
    
    train_loader  = DataLoader(train_dataset, batch_size=80, shuffle=True,  num_workers=8,
                               pin_memory=True, drop_last=False)
    valid_loader  = DataLoader(valid_dataset, batch_size=100, shuffle=False, num_workers=8,
                               pin_memory=True, drop_last=False)
    test_loader   = DataLoader(test_dataset,  batch_size=100, shuffle=False, num_workers=8,
                               pin_memory=True, drop_last=False)
    return train_loader, valid_loader, test_loader
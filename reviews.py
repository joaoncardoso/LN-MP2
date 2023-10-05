from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Read in the data from train.txt
with open("train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("test_just_reviews.txt", "r", encoding="utf-8") as f:
    lines_test = f.readlines()

# Define functions to handle the labels
def process(label): # labels can be "TRUTHFULPOSITIVE", "DECEPTIVEPOSITIVE",  "TRUTHFULNEGATIVE" or "DECEPTIVENEGATIVE"
    if label == "TRUTHFULPOSITIVE":
        return 0
    elif label == "DECEPTIVEPOSITIVE":
        return 1
    elif label == "TRUTHFULNEGATIVE":
        return 2
    elif label == "DECEPTIVENEGATIVE":
        return 3
    else:
        print("Error: label not found")
        return -1

def convert(label_int):
    if label_int == 0:
        return "TRUTHFULPOSITIVE"
    elif label_int == 1:
        return "DECEPTIVEPOSITIVE"
    elif label_int == 2:
        return "TRUTHFULNEGATIVE"
    elif label_int == 3:
        return "DECEPTIVENEGATIVE"
    else:
        print("Error: label not found")
        return -1

# Split the lines into labels and texts
train_labels = []
train_texts = []
for line in lines:
    label, text = line.strip().split("\t")
    train_labels.append(process(label))
    train_texts.append(text)

test_texts = []
for line in lines_test:
    text = line.strip()
    test_texts.append(text)

# train test split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

#  create a dataset class for the hotel reviews which inherits from torch Dataset

class HotelReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
train_dataset = HotelReviewsDataset(train_encodings, train_labels)
val_dataset = HotelReviewsDataset(val_encodings, val_labels)
test_dataset = HotelReviewsDataset(test_encodings, [0]*len(test_encodings['input_ids']))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        optim.zero_grad()
model.eval()


"""


from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        optim.zero_grad()
model.eval()


"""
# must convert this to problem domain pls copilot

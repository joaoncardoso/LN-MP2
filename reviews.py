from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from torch.optim import AdamW 
import os


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
val_dataset = HotelReviewsDataset(val_encodings, val_labels) # dataset for validation

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_dir = './model_save/'

model = None

if os.path.exists(model_dir):
    print("A saved model already exists in './model_save/'.")
    choice = input("Do you want to use the saved model (enter '1') or train a new DistilBert model (enter '2')? ")
    if choice == '1':
        # Load the saved model
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.train()
    elif choice == '2':
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        model.to(device)
        model.train()
    else:
        print("Invalid choice. Please enter '1' or '2'.")
else:
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.to(device)
    model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

# Get the number of labels in the dataset
num_labels = len(set(train_labels))

# Modify the model to match the number of labels in the dataset
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
model.to(device)
model.train()

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

# print accuracy on validation set

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)


# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
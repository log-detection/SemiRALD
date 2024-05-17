import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torch import nn, optim, torch
from sklearn.metrics import classification_report
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch import nn, optim, torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv("processed_data_HDFS.csv")  # File address and file name
df = df[['Processed_Content', 'Label']]
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])
df['Label'] = df['Label'].astype(int)
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_len = 128

# Dataset definition
class SentimentDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item] if self.labels is not None else None
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        data = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        if label is not None:
            data['labels'] = torch.tensor(label, dtype=torch.long)
        return data

# Model definition
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)
    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert =  RobertaModel.from_pretrained('roberta-base')
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.attention = Attention(256)
        self.classifier = nn.Linear(256 * 2, n_classes)
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        lstm_output, _ = self.lstm(bert_output)
        context_vector, _ = self.attention(lstm_output)
        output = self.classifier(context_vector)
        return output
# Model initialization
n_classes = df['Label'].nunique()
model = SentimentClassifier(n_classes=n_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
# Initial division of labeled and unlabeled data
initial_train_size = 0.01
df_labeled, df_unlabeled = train_test_split(df, test_size=1 - initial_train_size, stratify=df['Label'])
df_unlabeled.reset_index(drop=True, inplace=True)
train_data_loader = DataLoader(
    SentimentDataset(
        df_labeled['Processed_Content'].to_numpy(),
        df_labeled['Label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    ),
    batch_size=32,
    shuffle=True
)
# Training loop and self-training process
EPOCHS = 5
confidence_threshold = 0.90  # Confidence threshold
all_indexes = np.arange(len(df_unlabeled))  # Track the index of unlabeled data

# Adding plotting code in the training loop
for epoch in range(EPOCHS):
    # Initializing lists at the beginning of each epoch
    probs_list = []
    true_labels_list = []
    model.train()
    true_labels, pred_labels = [], []
    for data in train_data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, dim=1)
        pred_labels.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    print(f"Epoch {epoch + 1}:")
    print(classification_report(true_labels, pred_labels, digits=4))

    # Drawing confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Computing ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, pred_labels, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    # Self-training: Using the model to predict unlabeled data, converting high confidence results into new labeled data
    unlabeled_data_loader = DataLoader(
        SentimentDataset(
            df_unlabeled['Processed_Content'].to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len
        ),
        batch_size=64,
        shuffle=False
    )
    # Within the self-training loop
    model.eval()
    new_labels = []
    new_texts = []
    new_indexes = []
    with torch.no_grad():
        for i, data in enumerate(unlabeled_data_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probabilities, dim=1)
            confident = max_probs > confidence_threshold
            if confident.any():
                batch_indices = np.arange(len(data['input_ids']))
                confident_indices = batch_indices[confident.cpu().numpy()]
                global_indices = all_indexes[i * 64:(i + 1) * 64]
                selected_global_indices = global_indices[confident_indices]
                new_labels.extend(preds[confident].cpu().numpy())
                new_texts.extend(df_unlabeled.iloc[selected_global_indices]['Processed_Content'].tolist())
                new_indexes.extend(selected_global_indices)
        if new_labels:
            new_data = pd.DataFrame({
                'Processed_Content': new_texts,
                'Label': new_labels
            })
            df_labeled = pd.concat([df_labeled, new_data], ignore_index=True)
            df_unlabeled.drop(index=new_indexes, inplace=True)
            df_unlabeled.reset_index(drop=True, inplace=True)
            train_data_loader = DataLoader(
                SentimentDataset(
                    df_labeled['Processed_Content'].to_numpy(),
                    df_labeled['Label'].to_numpy(),
                    tokenizer=tokenizer,
                    max_len=max_len
                ),
                batch_size=64,
                shuffle=True
            )

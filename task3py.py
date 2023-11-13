import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Step 1: Load and Explore Data
df = pd.read_csv('D:/Internships/codesoft/TASK 3/Churn_Modelling.csv'
)
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

label_encoder = LabelEncoder()
df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

scaler = StandardScaler()
df[['CreditScore', 'Balance', 'EstimatedSalary']] = scaler.fit_transform(df[['CreditScore', 'Balance', 'EstimatedSalary']])

X = df.drop('Exited', axis=1).values
y = df['Exited'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define Enhanced Neural Network Model
class EnhancedNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.softmax(self.fc4(x), dim=1)
        return x

# Step 3: Prepare Data Loaders
input_dim = X_train.shape[1]
train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Step 4: Train the Enhanced Neural Network Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enhanced_NN_model = EnhancedNeuralNetwork(input_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(enhanced_NN_model.parameters(), lr=0.001, weight_decay=1e-5)

n_epochs = 20
for epoch in range(n_epochs):
    # Training loop
    enhanced_NN_model.train()
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = enhanced_NN_model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation loop
    enhanced_NN_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = enhanced_NN_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch+1}/{n_epochs}],Accuracy: {(100 * correct / total):.2f}% ,Test Accuracy: {(100 * correct / total):.2f}%')

print('Training Done ')

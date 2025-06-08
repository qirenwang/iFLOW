#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Custom Dataset class
class DriverDataset(Dataset):
   def __init__(self, features, labels):
       self.features = torch.FloatTensor(features)
       self.labels = torch.FloatTensor(labels)
       
   def __len__(self):
       return len(self.features)
   
   def __getitem__(self, idx):
       return self.features[idx], self.labels[idx]

# LSTM Model 
class LSTMModel(nn.Module):
   def __init__(self, input_size=2, hidden_size=45, num_classes=5):
       super(LSTMModel, self).__init__()
       self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.3)
       self.dropout = nn.Dropout(0.4)
       self.fc = nn.Linear(hidden_size, num_classes)
       self.softmax = nn.Softmax(dim=1)
       
   def forward(self, x):
       lstm_out, _ = self.lstm(x)
       lstm_out = lstm_out[:, -1, :]
       out = self.dropout(lstm_out)
       out = self.fc(out)
       out = self.softmax(out)
       return out

def train_model(model, train_loader, val_loader, num_epochs=100):
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   
   optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
   criterion = nn.CrossEntropyLoss()
   
   best_val_loss = float('inf')
   patience = 10
   patience_counter = 0
   
   train_losses = []
   val_losses = []
   
   for epoch in range(num_epochs):
       # Training
       model.train()
       train_loss = 0
       for batch_x, batch_y in train_loader:
           batch_x, batch_y = batch_x.to(device), batch_y.to(device)
           
           optimizer.zero_grad()
           outputs = model(batch_x)
           loss = criterion(outputs, batch_y)
           loss.backward()
           optimizer.step()
           train_loss += loss.item()
           
       avg_train_loss = train_loss / len(train_loader)
       train_losses.append(avg_train_loss)
           
       # Validation
       model.eval()
       val_loss = 0
       with torch.no_grad():
           for batch_x, batch_y in val_loader:
               batch_x, batch_y = batch_x.to(device), batch_y.to(device)
               outputs = model(batch_x)
               loss = criterion(outputs, batch_y)
               val_loss += loss.item()
               
       avg_val_loss = val_loss / len(val_loader)
       val_losses.append(avg_val_loss)
       
       # Print progress
       print(f'Epoch [{epoch+1}/{num_epochs}], '
             f'Train Loss: {avg_train_loss:.4f}, '
             f'Val Loss: {avg_val_loss:.4f}')
       
       # Learning rate adjustment
       scheduler.step(avg_val_loss)
       
       # Early stopping
       if avg_val_loss < best_val_loss:
           best_val_loss = avg_val_loss
           torch.save(model.state_dict(), 'best_model.pth')
           patience_counter = 0
       else:
           patience_counter += 1
           
       if patience_counter >= patience:
           print(f'Early stopping at epoch {epoch+1}')
           break
           
   # Plot training history
   plt.figure(figsize=(10,6))
   plt.plot(train_losses, label='Training Loss')
   plt.plot(val_losses, label='Validation Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Training History')
   plt.legend()
   plt.savefig('training_history.png')
   plt.close()
   
   return train_losses, val_losses

def main():
   # Load dataset
   trainset = np.load('/home/orin/mmfl/datasets/brain4cars/trainset.npy')
   trainsety = np.load('/home/orin/mmfl/datasets/brain4cars/trainsety.npy')
   testset = np.load('/home/orin/mmfl/datasets/brain4cars/testset.npy')
   testsety = np.load('/home/orin/mmfl/datasets/brain4cars/testsety.npy')

   # Reshape data
   trainset = np.reshape(trainset, (43130, 90, 2))
   trainsety = np.reshape(trainsety, (43130, 1, 1))
   testset = np.reshape(testset, (10329, 90, 2))
   testsety = np.reshape(testsety, (10329, 1, 1))

   # Convert labels to one-hot encoding
   newtrainsety = np.zeros((43130, 5))
   for i in range(43130):
       val1 = int(trainsety[i, 0, 0])
       newtrainsety[i, val1] = 1

   newtestsety = np.zeros((10329, 5))
   for i in range(10329):
       val1 = int(testsety[i, 0, 0])
       newtestsety[i, val1] = 1

   # Create datasets
   train_dataset = DriverDataset(trainset, newtrainsety)
   test_dataset = DriverDataset(testset, newtestsety)
   
   # Create dataloaders
   train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

   # Initialize and train model
   model = LSTMModel()
   train_losses, val_losses = train_model(model, train_loader, test_loader)
   
   # Save final model
   torch.save(model.state_dict(), 'final_model.pth')
   
   # Plot final results if needed
   plt.figure(figsize=(10,6))
   plt.plot(train_losses, label='Training Loss')
   plt.plot(val_losses, label='Validation Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Final Training History')
   plt.legend()
   plt.savefig('final_training_history.png')
   plt.close()

if __name__ == '__main__':
   main()
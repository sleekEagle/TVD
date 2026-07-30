from dataloaders import data_paths
from models import get_model
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import CONF
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim

def save_features(dataset='diving48', model_name='vjepa2'):
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model_name)

    out_path = os.path.join(CONF.OUT_PATH,'training', dataset, model_name)
    os.makedirs(out_path, exist_ok=True)
    features, labels = torch.empty(0), []
    for i in tqdm(range(len(path_list))):
        video = model.get_video(path_list[i])
        _=model.predict_video(video)
        f = model.get_features()
        features = torch.concatenate([features, f[None,:].cpu()],dim=0)
        cls = idx_list[i]
        labels.append(cls)
    np.save(os.path.join(out_path, 'features.npy'), features.numpy())
    np.save(os.path.join(out_path, 'labels.npy'), np.array(labels))

    # np.load(os.path.join(out_path, 'labels.npy')).shape



class FeatureDataset(Dataset):
    def __init__(self, path, prefix):
        features = np.load(os.path.join(path, f'{prefix}_features.npy'))
        labels = np.load(os.path.join(path, f'{prefix}_labels.npy'))
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        self.num_classes = len(np.unique(labels_encoded))
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

'''
Epoch [300/300], Train Loss: 1.5867, Train Acc: 0.5593, Val Loss: 2.0232, Val Acc: 0.4244

'''

class SingleLayerNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SingleLayerNN, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        # No activation function as we'll use CrossEntropyLoss which includes softmax
    
    def forward(self, x):
        return self.fc(x)

def save_checkpoint(model, optimizer, epoch, val_acc, val_loss, save_path='best_model.pth'):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, save_path)
    
    print(f"✓ Checkpoint saved to {save_path} (Val Acc: {val_acc:.4f})")

def train():
    epochs = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = r'D:\output\TVD\training\diving48\vjepa2'
    train_dataset = FeatureDataset(path, 'train')
    test_dataset = FeatureDataset(path, 'test')
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SingleLayerNN(input_size=1024, num_classes=48).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        #eval
        pass
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        if val_acc>best_val_acc:
            save_checkpoint(model, optimizer, epoch, val_acc, val_loss, save_path=os.path.join(path, 'best_model.pth'))
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')




    pass


if __name__ == "__main__":
    train()

    pass
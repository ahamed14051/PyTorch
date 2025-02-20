# Steps for GPU training
### 1. Check GPU Availability
```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
```
### 2. Move the Model to GPU
```python
model = MyModel()
model = model.to(device)#
```

### 3. Modify the training loop by moving data to GPU
```python
for batch_features, batch_labels in train_loader:
    batch_features, batch_labels = batch_features.to(device), batch_labels.to(device) #
```

### 4. Modify the evaluation loop too
```python
with torch.no_grad():
  for batch_features, batch_labels in test_loader:
    batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)#
```

###  5. Optimize the GPU Usage

To make the best use of GPU resources, apply the following optimizations:

#### a. Use Larger Batch Sizes
Larger batch sizes can better utilize GPU memory and reduce computation time per epoch (if memory allows).

```python
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True)
```
#### b. Enable DataLoader Pinning
```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
```

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import os.path as osp

def run_training(model, trainloader, validloader, epoch_, optimizer, scheduler, loss_fn, early_stopping_steps, verbose, device, fold, seed, path):
    
    early_step = 0
    best_loss = np.inf
    best_epoch = 0
    
    start = time.time()
    t = time.time() - start
    for epoch in range(epoch_):
        train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, device)
        valid_loss = valid_fn(model, loss_fn, validloader, device)
        # if ReduceLROnPlateau
        scheduler.step(valid_loss)
        if epoch % verbose==0 or epoch==epoch_-1:
            t = time.time() - start
            print(f"EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}, time: {t}")
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), osp.join( path, "weight",  f"{seed}_{fold}.pt") )
            early_step = 0
            best_epoch = epoch
        
        elif early_stopping_steps != 0:
            
            early_step += 1
            if (early_step >= early_stopping_steps):
                t = time.time() - start
                print(f"early stopping in iteration {epoch},  : best itaration is {best_epoch}, valid loss is {best_loss}, time: {t}")
                return model
    t = time.time() - start       
    print(f"training until max epoch {epoch_},  : best itaration is {best_epoch}, valid loss is {best_loss}, time: {t}")
    return model

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    cnt = 0
    for data in dataloader:
        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        inputs1, inputs2, targets = data['x'][0].to(device), data['x'][1].to(device), data['y'].to(device)
        outputs = model(inputs1, inputs2)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        # if cycle
        #scheduler.step()

        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs1, inputs2, targets = data['x'][0].to(device), data['x'][1].to(device), data['y'].to(device)
        outputs = model(inputs1, inputs2)
        loss = loss_fn(outputs, targets)
        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    return final_loss


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    for data in dataloader:
        inputs1, inputs2 = data['x'][0].to(device), data['x'][1].to(device)
        with torch.no_grad():
            outputs = model(inputs1, inputs2)
        preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds

def predict(model, testloader, device):
    model.to(device)
    predictions = inference_fn(model, testloader, device)
    
    return predictions

class MyDataset:
    def __init__(self, X, y, emb_cols):
        self.emb_cols = emb_cols
        self.numeric_cols = [col for col in X.columns if col not in self.emb_cols]
        self.emb_cols_idx = [X.columns.to_list().index(col) for col in self.emb_cols]
        self.numeric_cols_idx = [X.columns.to_list().index(col) for col in self.numeric_cols]

        self.X = X.values
        self.y = y.values
        
    def __len__(self):
        return (self.X.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : (torch.tensor(self.X[idx, self.emb_cols_idx], dtype=torch.long), torch.tensor(self.X[idx, self.numeric_cols_idx], dtype=torch.float)),
            'y' : torch.tensor(self.y[idx], dtype=torch.float),          
        }
        return dct
    
class TestDataset:
    def __init__(self, X, emb_cols):
        self.emb_cols = emb_cols
        self.numeric_cols = [col for col in X.columns if col not in self.emb_cols]
        self.emb_cols_idx = [X.columns.to_list().index(col) for col in self.emb_cols]
        self.numeric_cols_idx = [X.columns.to_list().index(col) for col in self.numeric_cols]

        self.X = X.values
        
    def __len__(self):
        return (self.X.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : (torch.tensor(self.X[idx, self.emb_cols_idx], dtype=torch.long), torch.tensor(self.X[idx, self.numeric_cols_idx], dtype=torch.float)),
        }
        return dct


def get_dataloader(X, y, emb_cols, batch_size, mode, model_type):
    if mode == "train":
        dataset = MyDataset(X, y, emb_cols)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    elif mode == "valid":
        dataset = MyDataset(X, y, emb_cols)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    elif mode == "test":
        dataset = TestDataset(X, emb_cols)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise ValueError

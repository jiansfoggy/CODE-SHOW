import torch
from torch import nn
import torch.nn.functional as F
from torcheval.metrics import BinaryAUROC, BinaryF1Score
from torcheval.metrics import BinaryConfusionMatrix, BinaryPrecision
from torcheval.metrics.classification import BinaryRecall
from tqdm import tqdm

# baseline part
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    acc_cnt = 0
    smp_cnt = 0
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, data in enumerate(tepoch):
            idx, text, label = data['input_ids'], data['attention_mask'], data['labels']
            idx, text, label = idx.to(device), text.to(device), label.to(device)
            y = model(idx,text)
            output = F.softmax(y, dim=1)
            pred = torch.argmax(output, dim=1)
            acc_cnt += pred.eq(label).sum()
            smp_cnt += len(label)
            loss = loss_fn(y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc = pred.eq(label).sum()/len(label)
            tepoch.set_postfix(loss=loss.item(), acc=train_acc.item())
    train_acc = (acc_cnt/smp_cnt).item()*100
    return train_acc
    
def valid(model, test_loader, loss_fn, device):
    model.eval()
    acc_cnt = 0
    smp_cnt = 0
    AUC, F1 = BinaryAUROC().to(device), BinaryF1Score().to(device)
    Confu_Matrix = BinaryConfusionMatrix().to(device)
    Pre, Rec = BinaryPrecision().to(device), BinaryRecall().to(device)
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for batch_idx, data in enumerate(test_loader):
                idx, text, label = data['input_ids'], data['attention_mask'], data['labels']
                idx, text, label = idx.to(device), text.to(device), label.to(device)
                y = model(idx,text)
                output = F.softmax(y, dim=1)
                pred = torch.argmax(output, dim=1)
                acc_cnt += pred.eq(label).sum()
                smp_cnt += len(label)
                loss = loss_fn(y, label)
                test_acc = pred.eq(label).sum() / len(label)
                tepoch.set_postfix(batch_ind=batch_idx, loss=loss.item(), acc=test_acc.item())
                F1.update(pred, label)
                AUC.update(pred, label)
                Confu_Matrix.update(pred, label)
                Pre.update(pred, label) 
                Rec.update(pred, label)
        test_acc = (acc_cnt/smp_cnt).item()*100
        test_AUC = AUC.compute().item()*100
        test_F1  = F1.compute().item()*100
        test_Pre = Pre.compute().item()*100
        test_Rec = Rec.compute().item()*100
        test_CM  = Confu_Matrix.compute()
        print("Valid Acc: %.2f%%, F1: %.2f%%, AUC: %.2f%%" % (test_acc, test_F1, test_AUC))
        print("Valid Precision: %.2f%%, Recall: %.2f%%" % (test_Pre, test_Rec))
        print("Valid Confusion Matrix: ", test_CM)
    return test_acc, test_F1, test_AUC, test_Pre, test_Rec, test_CM

def test(model, test_loader, loss_fn, device):
    acc_cnt = 0
    smp_cnt = 0
    AUC, F1 = BinaryAUROC().to(device), BinaryF1Score().to(device)
    Confu_Matrix = BinaryConfusionMatrix().to(device)
    Pre, Rec = BinaryPrecision().to(device), BinaryRecall().to(device)
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for batch_idx, data in enumerate(test_loader):
                idx, text, label = data['input_ids'], data['attention_mask'], data['labels']
                idx, text, label = idx.to(device), text.to(device), label.to(device)
                y = model(idx,text)
                output = F.softmax(y, dim=1)
                pred = torch.argmax(output, dim=1)
                acc_cnt += pred.eq(label).sum()
                smp_cnt += len(label)
                loss = loss_fn(y, label)
                test_acc = pred.eq(label).sum() / len(label)
                tepoch.set_postfix(batch_ind=batch_idx, loss=loss.item(), acc=test_acc.item())
                F1.update(pred, label)
                AUC.update(pred, label)
                Confu_Matrix.update(pred, label)
                Pre.update(pred, label) 
                Rec.update(pred, label)
        test_acc = (acc_cnt/smp_cnt).item()*100
        test_AUC = AUC.compute().item()*100
        test_F1  = F1.compute().item()*100
        test_Pre = Pre.compute().item()*100
        test_Rec = Rec.compute().item()*100
        test_CM  = Confu_Matrix.compute()
        print("Test Acc: %.2f%%, F1: %.2f%%, AUC: %.2f%%" % (test_acc, test_F1, test_AUC))
        print("Test Precision: %.2f%%, Recall: %.2f%%" % (test_Pre, test_Rec))
        print("Test Confusion Matrix: ", test_CM)
    return test_acc, test_F1, test_AUC, test_Pre, test_Rec, test_CM
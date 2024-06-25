import argparse, faulthandler, math, torch
import numpy as np
import torch.nn as nn
from transformers import RobertaModel
from utils.data_process import get_dataloader
from utils.train_val import train, valid, test
from model.RoBERTaCNN import RoBERTaCNN

torch.backends.cudnn.deterministic = True
faulthandler.enable()
torch.manual_seed(42)
torch.cuda.empty_cache()
use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"

def main():
    parser = argparse.ArgumentParser(description='PyTorch Protein Sequence Activation Status')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    file_dir = '../file_path.xlsx'
    train_loader, valid_loader, test_loader = get_dataloader(file_dir, max_len=128)
    all_num_set    = len(train_loader) + len(test_loader)
    step_size_up   = math.floor(0.65 * all_num_set)
    step_size_down = all_num_set - step_size_up

    roberta = RobertaModel.from_pretrained('roberta-base', num_labels=2)
    model = RoBERTaCNN(roberta, num_classes=2).to(device)
    paras = filter(lambda p: p.requires_grad, model.parameters())
    paras = sum([np.prod(p.size()) for p in paras]) / 1_000_000
    print('Trainable Parameters: %.3fM' % paras)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, gamma=0.85,
                                                  step_size_up=step_size_up, step_size_down=step_size_down,
                                                  mode='triangular2', cycle_momentum=False)
    best_acc = 0
    record = open("./exp_record_path.txt", 'a')
    record.write("Epoch Train_Acc Valid_Acc F1_Score AUC Precision Recall Conf_Mat" + "\n")
    record.write("="*50 + "\n")
    record.close()
    for epoch in range(1, 61):
        print(f'Epoch {epoch}/60:')
        train_acc = train(model, train_loader, optimizer, loss_fn, device)
        valid_acc, F1, AUC, Pre, Rec, CM = valid(model, valid_loader, loss_fn, device)
        scheduler.step()
        
        if valid_acc > best_acc:
            PATH = './weight_file_path.h5'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, PATH)
            best_acc = valid_acc
        
        record = open("./exp_record_path.txt", 'a')
        record.write(str(epoch) + " " + str(train_acc) + " " + str(valid_acc) + " " 
                     + str(F1) + " " + str(AUC) + " " + str(Pre) + " " + str(Rec)
                     + "\n")
        record.write(str(CM) + "\n")
        record.close()
    
    checkpoint = torch.load('./weight_file_path.h5')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    test_acc, F1, AUC, Pre, Rec, CM = test(model, test_loader, loss_fn, device)
    record = open("./exp_record_path.txt", 'a')
    record.write("="*50 + "\n")
    record.write("Test_Acc F1_Score AUC Precision Recall Conf_Mat" + "\n")
    record.write("="*50 + "\n")
    record.write(str(test_acc) + " " + str(F1) + " " + str(AUC) + " " + str(Pre) 
                 + " " + str(Rec) + "\n")
    record.write(str(CM) + "\n")
    record.write("="*50 + "\n")
    record.close()
if __name__ == '__main__':
	main()
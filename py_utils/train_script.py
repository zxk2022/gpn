import torch
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import csv
import time
import os
from sklearn.metrics import accuracy_score, precision_score


def evaluate(loader, use_mp, keys, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, use_mp, keys).max(dim=1)[1]
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    # 计算 OA
    oa = accuracy_score(all_labels, all_preds)
    
    # 计算 MACC
    macc = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return oa, macc

def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, device, scheduler, args, num_exp, use_mp_tr=True, use_mp_val=True, use_mp_te=True):
    writer = SummaryWriter(f'{args.output}/logs/exp{num_exp}')
    best_val_accuracy = 0
    best_val_macc = 0
    best_model_state = None

    chosen_trainlog = {}

    total_time = 0

    for epoch in tqdm(range(args.epochs), f"Training Progress Of Exp{num_exp}"):
        model.train()
        train_correct = 0
        train_total = 0
        start_time = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch, use_mp=use_mp_tr, keys=args.keys, undirected=args.undirected)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            # 计算训练精度
            pred = out.max(dim=1)[1]
            train_correct += pred.eq(batch.y).sum().item()
            train_total += batch.y.size(0)
        scheduler.step()
        end_time = time.time()
        total_time += end_time - start_time

        # 每轮训练结束后进行验证
        val_oa, val_macc = evaluate(val_loader, use_mp_val, args.keys, model, device)
        train_accuracy = train_correct / train_total
        # 使用writer.add_scalar记录标量数据
        writer.add_scalar('Loss/Train', loss.item(), epoch)
        writer.add_scalar('Accuracy/TrainOA', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Validation/OA', val_oa, epoch)
        writer.add_scalar('Accuracy/Validation/MACC', val_macc, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # 保存验证精度最高的模型
        if val_oa > best_val_accuracy or (val_oa == best_val_accuracy and val_macc > best_val_macc):
            best_val_accuracy = val_oa
            best_val_macc = val_macc
            chosen_trainlog = {
                'Epoch': epoch,
                'Loss': loss.item(),
                'Train Accuracy': train_accuracy,
                'Val OA': val_oa,
                'Val MACC': val_macc
            }
            torch.save(model.state_dict(), f'{args.output}/logs/exp{num_exp}/best_model.pt')

    writer.close()
    print(f'Total Training Time: {total_time} seconds')

    # 保存最后一轮的模型
    torch.save(model.state_dict(), f'{args.output}/logs/exp{num_exp}/last_model.pt')
    # 使用验证精度最高的模型进行测试
    model.load_state_dict(torch.load(f'{args.output}/logs/exp{num_exp}/best_model.pt'))
    test_oa, test_macc = evaluate(test_loader, use_mp_te, args.keys, model, device)
    model.load_state_dict(torch.load(f'{args.output}/logs/exp{num_exp}/last_model.pt'))
    final_test_oa, final_test_macc = evaluate(test_loader, use_mp_te, args.keys, model, device)

    return {
        'Best Validation OA': best_val_accuracy,
        'Best Validation MACC': best_val_macc,
        'Chosen Epoch': chosen_trainlog['Epoch'],
        'Chosen Loss': chosen_trainlog['Loss'],
        'Chosen Train Accuracy': chosen_trainlog['Train Accuracy'],
        'Chosen Val OA': chosen_trainlog['Val OA'],
        'Chosen Val MACC': chosen_trainlog['Val MACC'],
        'Final Epoch Val OA': val_oa,
        'Final Epoch Val MACC': val_macc,
        'Test OA': test_oa,
        'Test MACC': test_macc,
        'Final Epoch Test OA': final_test_oa,
        'Final Epoch Test MACC': final_test_macc,
        'Total Time': total_time
    }

def save_results_to_csv(result, filename):
    # 定义字段名
    fields = [
        'Best Validation OA', 'Best Validation MACC', 
        'Chosen Epoch', 'Chosen Loss', 'Chosen Train Accuracy', 
        'Chosen Val OA', 'Chosen Val MACC', 
        'Final Epoch Val OA', 'Final Epoch Val MACC',
        'Test OA', 'Test MACC', 
        'Final Epoch Test OA', 'Final Epoch Test MACC',
        'Total Time'
    ]
    
    # 检查文件是否已经存在
    file_exists = os.path.exists(filename)
    
    # 打开文件，如果不存在则创建
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        
        # 如果文件是新创建的，写入字段名
        if not file_exists:
            writer.writeheader()
        
        # 写入结果
        writer.writerow(result)
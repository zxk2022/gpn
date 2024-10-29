import torch
from thop import profile

def measure_model(model, train_loader, args, use_mp_tr=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 计算参数数量
    params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {params}')

    # 从train_dataloader中获取一个data数据
    input_data = next(iter(train_loader))[0].to(device)  # 假设返回的是一个包含输入数据的元组
    # 创建输入参数的元组
    input_data = (input_data, use_mp_tr, args.keys, args.undirected,)

    # 使用thop库计算FLOPs，同时传递额外的参数
    flops, _ = profile(model, inputs=input_data)
    print(f'FLOPs: {flops}')

    # 初始化batch_size和total_iters
    batch_size = train_loader.batch_size
    total_iters = len(train_loader)//batch_size

    # 计算吞吐量
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)  # 同样假设这里是一个元组
            _ = model(batch, use_mp=use_mp_tr, keys=args.keys, undirected=args.undirected)
    end_time.record()
    torch.cuda.synchronize()

    # 计算总时间（毫秒）
    total_time_ms = start_time.elapsed_time(end_time)
    # 转换为秒
    total_time_s = total_time_ms / 1000
    # 总样本数
    total_samples = len(train_loader.dataset)
    # 计算吞吐量
    throughput = total_samples / total_time_s
    print(f'Throughput: {throughput:.2f} samples/sec')
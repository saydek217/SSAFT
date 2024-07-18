import argparse
import copy
import csv
import os
import warnings
import numpy
import torch
import tqdm
import yaml
from torch.utils import data
from nets import nn
from utils import util
from utils.dataset import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
tb_writer = SummaryWriter() #create the log_file 
import cv2

warnings.filterwarnings("ignore")
def learning_rate(args, initial_lrf=0.01, final_lrf=0.001):
    def fn(x):
        if x <= 80:
            # Phase 1: Initial part towards an lrf of 0.01
            return (1 - x / 100) * (1.0 - initial_lrf) + initial_lrf
        else:
            # Phase 2: Transition from 0.01 to 0.001
            start_lr_phase_2 = initial_lrf  # Starting LR for the second phase
            end_lr_phase_2 = final_lrf  # Ending LR for the second phase
            # Adjust the rate for the second phase
            return start_lr_phase_2 + (x - 50) / (args['epochs'] - 50) * (end_lr_phase_2 - start_lr_phase_2)
    return fn

#Initilization
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
 
def train(args, params, initial_lrf=0.01, final_lrf=0.001):
    # Model
    model = nn.yolo_v8_n(len(params['names'].values())).cuda()
    model.apply(init_weights)

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64
    p = [], [], []
    
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)

    optimizer = torch.optim.SGD(p[2], params['lr0'], params['momentum'], nesterov=True)
    optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': p[1]})
    del p

    # Scheduler
    lr_lambda=learning_rate(args, initial_lrf, final_lrf)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    
    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
            
    with open('../data/train.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append('../data/new_data/train/' + filename)

    dataset = Dataset(filenames, args.input_size, params, True)
    if args.world_size <= 1:
        sampler = None
    else:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler, num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)


    if args.world_size > 1:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)

    # Start training
    best = 0
    num_batch = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)
    learning_rates = []
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)
    checkpoint_path = './weights/best.pt'
        # Resume training if a checkpoint exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best = checkpoint['best_mAP']
        loss_history = checkpoint['loss_history']
        learning_rates = checkpoint['learning_rates']
     
        # Other components you may have saved  
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        
    with open('weights/step.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'mAP@50', 'mAP'])
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            model.train()
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            m_loss = util.AverageMeter()
            if args.world_size > 1:
                sampler.set_epoch(epoch)
            p_bar = enumerate(loader)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
            if args.local_rank == 0:
                p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

            optimizer.zero_grad()

            for i, (samples, targets, _) in p_bar:
                x = i + num_batch * epoch  # number of iterations
                samples = samples.cuda().float() / 255
                targets = targets.cuda()
            
                # Warmup
                
                if x <= num_warmup:
                    #warmup_factor = x / float(num_warmup)
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, numpy.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [params['warmup_bias_lr'], 0.01 * 0.01]
                        else:
                            fp = [0.0, 0.01 * 0.01]
                        y['lr'] = numpy.interp(x, xp, fp)
                        if 'momentum' in y:
                            fp = [params['warmup_momentum'], params['momentum']]
                            y['momentum'] = numpy.interp(x, xp, fp)

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)  # forward
                loss = criterion(outputs, targets)

                m_loss.update(loss.item(), samples.size(0))

                loss *= args.batch_size  # loss scaled by batch_size
                loss *= args.world_size  # gradient averaged between devices in DDP mode
                tb_writer.add_scalar('Training Loss/total', m_loss.avg, global_step=epoch) # log the loss

                # Backward
                amp_scale.scale(loss).backward()

                # Optimize
                if x % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.4g') % (f'{epoch + 1}/{args.epochs}', memory, m_loss.avg)
                    p_bar.set_description(s)

                del loss
                del outputs

            # Scheduler
            scheduler.step()

            # Store the learning rate for plotting
            
            current_lr = scheduler.optimizer.param_groups[0]['lr']  # Get the current learning rate
            learning_rates.append(current_lr)
            tb_writer.add_scalar('Learning rate ', current_lr, global_step=epoch) # log the learning rate 
            
            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)
                tb_writer.add_scalar('mAP', last[1], global_step=epoch)
                tb_writer.add_scalar('mAP@50', last[0], global_step=epoch)
                tb_writer.add_scalar('Precision', last[2], global_step=epoch)
                tb_writer.add_scalar('recall', last[3], global_step=epoch)
                writer.writerow({'mAP': str(f'{last[1]:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3),
                                 'mAP@50': str(f'{last[0]:.3f}')})
                f.flush()

                # Update best mAP
                if last[1] > best:
                    best = last[1]

                # Save model
                checkpoint = {
                    'epoch': epoch,
                    'model': copy.deepcopy(ema.ema).half(),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mAP': best,
                    'loss_history': m_loss,
                    'learning_rates': learning_rates
                }


                # Save last, best and delete
                torch.save(checkpoint, './weights/last.pt')
                if best == last[1]:
                    torch.save(checkpoint, './weights/best.pt')
                del checkpoint

    torch.cuda.empty_cache()

@torch.no_grad()
def test(args, params, model=None):
    filenames = []
    
    with open('../data/val.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append('../data/new_data/val/' + filename)

    dataset = Dataset(filenames, args.input_size, params, False)
    loader = data.DataLoader(dataset, 8, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    if model is None:
        model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()

    model.half()
    model.eval()
    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))
    for samples, targets, shapes in p_bar:
        samples = samples.cuda()
        targets = targets.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        _, _, height, width = samples.shape  # batch size, channels, height, width

        # Inference
        outputs = model(samples)
        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
        outputs = util.non_max_suppression(outputs, 0.001, 0.65)

        # Metrics
        for i, output in enumerate(outputs):
            labels = targets[targets[:, 0] == i, 1:]
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if labels.shape[0]:
                    metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                continue

            detections = output.clone()
            util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

            # Evaluate
            if labels.shape[0]:
                tbox = labels[:, 1:5].clone()  # target boxes
                tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                correct = numpy.zeros((detections.shape[0], iou_v.shape[0]))
                correct = correct.astype(bool)

                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                correct_class = t_tensor[:, 0:1] == detections[:, 5]
                for j in range(len(iou_v)):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                        matches = matches.cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
            metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)

    # Print results
    print('%10.3g' * 3 % (m_pre, m_rec, mean_ap))

    # Return results
    model.float()  # for training
    return map50, mean_ap, m_pre, m_rec

args = {
    'input_size': 640,
    'batch_size': 16,
    'local_rank': 0,  # This might be irrelevant in a non-distributed setup
    'epochs': 200,
    'train': True,  # Set to False if you don't want to train
    'test': False,  # Set to True if you want to test
    'world_size': 1 # Assuming a single-process setup
}

# Adjust for potential distributed computing environments, even though it might not be applicable
args['local_rank'] = int(os.getenv('LOCAL_RANK', 0))
args['world_size'] = int(os.getenv('WORLD_SIZE', 1))

if args['world_size'] > 1:
    torch.cuda.set_device(device=args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

if args['local_rank'] == 0:
    if not os.path.exists('weights'):
        os.makedirs('weights')

# Assuming util is a module with these functions. If not, you'll need to define them or adjust accordingly.
util.setup_seed()
util.setup_multi_processes()

# Load parameters from args.yaml
with open('./utils/args_indoor_openImages.yaml', 'r') as f:
    params = yaml.safe_load(f)
args_namespace = argparse.Namespace(**args)

train(args_namespace,params)
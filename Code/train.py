# Lib
import torch
from get_data import ImageDataset
from model import TCRnet
import torch.nn as nn
import os
import sys
import util
import time
from torch.utils.tensorboard import SummaryWriter


# Main
def main(args):
    # Define global variable
    root_dir = os.path.dirname(os.path.dirname(__file__))
    base_file_name = util.base_file_name(args)
    out_root = os.path.join(root_dir, 'Out')
    checkp_dir = os.path.join(out_root+'/Checkpoint', base_file_name)
    summary_dir = os.path.join(out_root+'/Tensorboard', base_file_name)
    log_dir = out_root + '/Log/' + base_file_name + '.log'
    best_acc = 0.0
    counter = 0
    args.summary_dir = summary_dir

    # Log output
    sys.stdout = util.Logger(log_dir, sys.stdout) # redirect std output
    sys.stderr = util.Logger(log_dir, sys.stderr) # redirect std err, if necessary

    # Devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # Dataset
    if args.dataset == 'RAF-DB':
        args.num_classes = 7
        # subset
        if args.subset == 'base':
            img_dir = os.path.join(root_dir, 'Data/RAF-DB/Aligned')
            train_txt_dir = os.path.join(root_dir,'Text/RAF-DB/base_train.txt')
            val_txt_dir = os.path.join(root_dir,'Text/RAF-DB/base_val.txt')

    # Model
    # choose model type
    if 'TCR' in args.model:
        model = TCRnet(num_classes=args.num_classes, pool_type=args.pool_type,
                        num_heads=args.num_heads, blocks=args.blocks, bias=args.bias, 
                        dropout=args.dropout, model_type=args.model)
    # parallel computer
    model = torch.nn.DataParallel(model).cuda()
    # pretrain
    if args.pretrain == 'MS_Celeb_1M': 
        checkpoint = torch.load('/148Dataset/data-wen.yaoli/CODE/RAN/data/checkpoint/ijba_res18_naive.pth.tar')
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                pass # 去掉全连接层
            else:    
                model_state_dict[key] = pretrained_state_dict[key]
        model.load_state_dict(model_state_dict, strict = False)

    # Dataloader
    train_dataset =  ImageDataset(img_dir=img_dir, txt_dir=train_txt_dir,
                                dataset_name=args.dataset, transform=None) # tensor [3,224,224] 
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.mini_batch, shuffle=True,
            num_workers=args.workers, pin_memory=True) # 训练集dataloader

    val_dataset =  ImageDataset(img_dir=img_dir, txt_dir=val_txt_dir,
                                dataset_name=args.dataset, transform=None) # tensor [3,224,224] 
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.mini_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True) # 验证集dataloader

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr ,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay) 
                                    # v(t+1) = m*v(t) + g(t+1)  p(t+1) = p(t) - lr*v(t+1)   weight decay (L2 penalty)

    # Epoch
    for epoch in range(args.epochs):
        # display split line 
        print('{:#^75}'.format(' Epoch {} '.format(epoch)))

        # 调整学习率
        util.adjust_learning_rate(args=args, optimizer=optimizer, epoch=epoch) # 调整学习率

        # train & validate
        counter = train(args, train_loader, model, criterion, optimizer, epoch, counter)
        prec_acc = validate(args, val_loader, model, criterion, epoch)

        # save best model
        is_best = prec_acc > best_acc
        best_acc = max(prec_acc, best_acc)
        if is_best:
            best_epoch = epoch
        util.save_checkpoint(state={
            'epoch': epoch,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best=is_best.item(), checkp_dir=checkp_dir)

        # display best accturacy
        print("Best Accuracy Is {:0<6.4f}% (epoch:{})".format(best_acc*100, best_epoch))


#  Train
def train(args, train_loader, model, criterion, optimizer, epoch, counter):
    start = time.time()
    mini_steps = args.batch_size // args.mini_batch
    prec_acc = util.MatrixMeter(args.num_classes)
    loss_data = util.AverageMeter()
    base_loss_data = util.AverageMeter()
    batch_num = len(train_loader)-1
    record_interval = (batch_num // 50 + 1) * 10

    # switch to train mode
    model.train()

    # train one epoch
    optimizer.zero_grad()
    for i, (img, label) in enumerate(train_loader):
        mini_batch_size = label.size(0)

        # Forward propagation
        input_var = torch.autograd.Variable(img.cuda()) # [B, 3, 224, 224] 放入变量图中
        target_var = torch.autograd.Variable(label.cuda()) # [B,1] 放入变量图中
        prec_score, _ = model(input_var)

        # Loss
        base_loss = criterion(prec_score, target_var) # crossentropy的均值
        loss = base_loss

        # Back propagation
        loss_mini = loss * mini_batch_size / args.batch_size
        loss_mini.backward()
        if (i+1) % mini_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            optimizer.zero_grad()

        # Update data
        prec_acc.update(prec_score.cpu(), label.cpu())
        loss_data.update(loss.item())
        base_loss_data.update(base_loss.item())

        # Display and save value result
        if i % record_interval == 0:
            print(
                '{:<12}{:<12}{:<21}{}'.format('[{:0>3}/{}]'.format(i, batch_num),
                 'epoch:{:0>3}'.format(epoch),'loss:{:0<13.11f}'.format(loss_data.val())
                , 'acc:{:0<8.6f}'.format(prec_acc.val_acc()))
            )
            with SummaryWriter(args.summary_dir) as writer: 
                writer.add_scalar('Val_trian/base_loss', base_loss_data.val(), counter)
                writer.add_scalar('Val_trian/Loss', loss_data.val(), counter)
                writer.add_scalar('Val_trian/Acc', prec_acc.val_acc(), counter)
            counter += 1
        
    # Final grad update
    nn.utils.clip_grad_norm_(model.parameters(), 10.)
    optimizer.step()
    optimizer.zero_grad()

    # Display and save avg result
    with SummaryWriter(args.summary_dir) as writer: 
        writer.add_scalar('Avg_trian/base_loss', base_loss_data.avg(), epoch)
        writer.add_scalar('Avg_trian/Loss', loss_data.avg(), epoch)
        writer.add_scalar('Avg_trian/Acc', prec_acc.avg_acc(), epoch)
        for i, lab_acc in enumerate(prec_acc.label_acc()):
            writer.add_scalar('Avg_trian_label/acc'+str(i), lab_acc, epoch)
        heatmap = util.heatmap(prec_acc.confus_matrix(), args.dataset)
        writer.add_figure(tag='Train Confusion Matrix',
                          figure=heatmap, global_step=epoch)
    print('Train [{:0>3}]  Loss:{:0<13.11f}  Acc:{:0<8.6f}  Time:{:.2f}'.format(
            epoch, loss_data.val(), prec_acc.avg_acc(), time.time()-start))
    return counter


# Validate
def validate(args, val_loader, model, criterion, epoch):
    with torch.no_grad():
        start = time.time()
        prec_acc = util.MatrixMeter(args.num_classes)
        loss_data = util.AverageMeter()
        base_loss_data = util.AverageMeter()

        # switch to evaluate mode
        model.eval()

        for img, label in val_loader:
            # Forward propagation
            input_var = torch.autograd.Variable(img.cuda()) # [B, 3, 224, 224] 放入变量图中
            target_var = torch.autograd.Variable(label.cuda()) # [B,1] 放入变量图中
            prec_score, _ = model(input_var)

            # Loss
            base_loss = criterion(prec_score, target_var) # crossentropy的均值
            loss = base_loss

            # Update data
            prec_acc.update(prec_score.cpu(), target_var.cpu())
            loss_data.update(loss.item())
            base_loss_data.update(base_loss.item())

        # Display and save avg result 
        with SummaryWriter(args.summary_dir) as writer: 
            writer.add_scalar('Avg_test/base_loss', base_loss_data.avg(), epoch)
            writer.add_scalar('Avg_test/Loss', loss_data.avg(), epoch)
            writer.add_scalar('Avg_test/Acc', prec_acc.avg_acc(), epoch)
            for i, lab_acc in enumerate(prec_acc.label_acc()):
                writer.add_scalar('Avg_test_label/acc'+str(i), lab_acc, epoch)
            heatmap = util.heatmap(prec_acc.confus_matrix(), args.dataset)
            writer.add_figure(tag='Test Confusion Matrix',
                              figure=heatmap, global_step=epoch)
        print("Test  [{:0>3}]  Loss:{:0<13.11f}  Acc:{:0<8.6f}  Time:{:.2f}".format(
                epoch, loss_data.val(), prec_acc.avg_acc(), time.time()-start))
    return prec_acc.avg_acc()
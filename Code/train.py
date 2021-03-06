# Lib
import torch
from get_data import ImageDataset
from model import TCRnet
import torch.nn as nn
import os
import sys
import util
import time
import loss_fn
from torch.utils.tensorboard import SummaryWriter


# Main
def main(args):
    # Define global variable
    root_dir = os.path.dirname(os.path.dirname(__file__))
    base_file_name = util.base_file_name(args)
    out_root = os.path.join(root_dir, 'Out/base')
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
        # RAF-DB subset
        if args.subset == 'base':
            img_dir = os.path.join(root_dir, 'Data/RAF-DB/Aligned')
            train_txt_dir = os.path.join(root_dir,'Text/RAF-DB/base_train.txt')
            val_txt_dir = os.path.join(root_dir,'Text/RAF-DB/base_val.txt')
        if args.subset == 'flip':
            img_dir = os.path.join(root_dir, 'Data/RAF-DB/Aligned')
            train_txt_dir = os.path.join(root_dir,'Text/RAF-DB/flip_train.txt')
            val_txt_dir = os.path.join(root_dir,'Text/RAF-DB/base_val.txt')

    # Model
    # choose model type
    if 'TCR' in args.model:
        model = TCRnet(num_classes=args.num_classes,
                        local_start=args.local_start, radio=tuple(args.radio), patch_num=tuple(args.patch_num),
                        trans_layer=args.trans_layer, res=args.res, pool_type=args.pool_type,
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
                pass # ??????????????????
            else:    
                model_state_dict[key] = pretrained_state_dict[key]
        model.load_state_dict(model_state_dict, strict = False)
        print('pretrained by MS_Celeb_1M')

    # Dataloader
    train_dataset =  ImageDataset(img_dir=img_dir, txt_dir=train_txt_dir,
                                dataset_name=args.dataset, transform=None) # tensor [3,224,224] 
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.mini_batch, shuffle=True,
            num_workers=args.workers, pin_memory=True) # ?????????dataloader

    val_dataset =  ImageDataset(img_dir=img_dir, txt_dir=val_txt_dir,
                                dataset_name=args.dataset, transform=None) # tensor [3,224,224] 
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.mini_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True) # ?????????dataloader

    # Criterion
    criterion = {}
    criterion['base'] = nn.CrossEntropyLoss().cuda()
    if args.var_loss:
        criterion['var'] = loss_fn.VarLoss().cuda()

    # Optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr ,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay) 
                                    # v(t+1) = m*v(t) + g(t+1)  p(t+1) = p(t) - lr*v(t+1)   weight decay (L2 penalty)

    # Epoch
    for epoch in range(args.epochs):
        # display split line 
        print('{:#^120}'.format(' Epoch {} '.format(epoch)))

        # ???????????????
        util.adjust_learning_rate(args=args, optimizer=optimizer, epoch=epoch) # ???????????????

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
        print('{:_<75}'.format(''))
        print('\033[1;36m'
        "Best Accuracy Is {:0<6.4f}% (epoch:{})"
        '\033[0m'.format(best_acc*100, best_epoch))


#  Train
def train(args, train_loader, model, criterion, optimizer, epoch, counter):
    start = time.time()
    mini_steps = args.batch_size // args.mini_batch
    # total
    prec_acc = util.MatrixMeter(args.num_classes)
    loss_data = util.AverageMeter()
    # global
    prec_acc_gl = util.MatrixMeter(args.num_classes)
    base_loss_gl_data = util.AverageMeter()
    var_loss_data = util.AverageMeter()
    # local
    prec_acc_lc = util.MatrixMeter(args.num_classes)
    patch_acc = util.MatrixMeter(args.num_classes, args.patch_num[0]*args.patch_num[1])
    base_loss_lc_all_data = util.AverageMeter()
    base_loss_lc_data = util.AverageMeter()
    var_loss_lc_data = util.AverageMeter()
    # constant
    batch_num = len(train_loader)-1
    record_interval = (batch_num // 50 + 1) * 10

    # switch to train mode
    model.train()

    # train one epoch
    optimizer.zero_grad()
    for i, (img, label) in enumerate(train_loader):
        mini_batch_size = label.size(0)

        # Forward propagation
        input_var = torch.autograd.Variable(img.cuda()) # [B, 3, 224, 224] ??????????????????
        target_var = torch.autograd.Variable(label.cuda()) # [B,1] ??????????????????
        total_score, score_gl, score_lc, score_lc_all, attention_gl, attention_lc = model(input_var)

        # Loss
        ## global
        ### base loss
        base_loss_gl = criterion['base'](score_gl, target_var) # cross entropy?????????
        loss = base_loss_gl
        ### var loss
        if args.var_loss:
            var_loss_gl = criterion['var'](attention_gl)
            loss += args.var_rate * var_loss_gl.sum()

        ## local
        if args.local_start > 0:
            ### base loss
            base_loss_lc_all = []
            for scl in score_lc_all.transpose(0,1):
                base_loss_lc_all.append(criterion['base'](scl, target_var)) # ??????local branch???cross entropy?????????
            base_loss_lc_all = torch.stack(base_loss_lc_all, dim=0) # [P] ??????local ???????????????base loss
            ### (avg) base loss
            base_loss_lc = criterion['base'](score_lc, target_var) # ???????????????cross entropy??????
            loss += base_loss_lc
            ### var loss
            if args.var_loss:
                var_loss_lc = criterion['var'](attention_lc)
                loss = loss+args.var_rate_lc * var_loss_lc
        ### default
        else:
            base_loss_lc_all = torch.Tensor([0]).cuda()
            base_loss_lc = torch.Tensor([0]).cuda()
            var_loss_lc = torch.Tensor([0]).cuda()

        # Back propagation
        loss_mini = loss * mini_batch_size / args.batch_size
        loss_mini.backward()
        if (i+1) % mini_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            optimizer.zero_grad()

        # Update data
        ## total
        prec_acc.update(total_score.cpu(), label.cpu())
        loss_data.update(loss.item())
        ## global
        prec_acc_gl.update(score_gl.cpu(), label.cpu())
        base_loss_gl_data.update(base_loss_gl.item())
        var_loss_data.update(var_loss_gl.detach().cpu().numpy() * args.var_rate)
        ## local
        if args.local_start > 0:
            prec_acc_lc.update(score_lc.cpu(), label.cpu())
            patch_acc.update(score_lc_all.cpu(), label.cpu())
            base_loss_lc_all_data.update(base_loss_lc_all.detach().cpu().numpy())
            base_loss_lc_data.update(base_loss_lc.item())
            var_loss_lc_data.update(var_loss_lc.item() * args.var_rate_lc)

        # Display and save value result
        if i % record_interval == 0:
            print(
                '\033[0;{num}m'
                '{progress:<12}{epoch:<12}{acc:<16}{loss:<21}\n'
                '{gl:<12}{base_loss_gl:<26}{var_gl:<25}{acc_gl:<16}\n'
                '{lc:<12}{base_loss_lc:<26}{var_lc:<25}{acc_lc:<16}'
                '\033[0m'.format(
                num=i%3+33,
                progress='[{:0>3}/{}]'.format(i, batch_num),
                epoch='epoch:{:0>3}'.format(epoch), 
                loss='loss:{:0<13.11f}'.format(loss_data.val()),
                acc='acc:{:0<8.6f}'.format(prec_acc.val_acc()),
                gl='[global]',
                base_loss_gl='base loss:{:0<13.11f}'.format(base_loss_gl_data.val()),
                var_gl='var loss:{:0<13.11f}'.format(var_loss_data.val().sum()),
                acc_gl='acc:{:0<8.6f}'.format(prec_acc_gl.val_acc()),
                lc='[local]',
                base_loss_lc='base loss:{:0<13.11f}'.format(base_loss_lc_data.val()),
                var_lc='var loss:{:0<13.11f}'.format(var_loss_lc_data.val()),
                acc_lc='acc:{:0<8.6f}'.format(prec_acc_lc.val_acc()) 
            )
            )
            with SummaryWriter(args.summary_dir) as writer: 
                writer.add_scalar('Val_trian/base_loss_gl', base_loss_gl_data.val(), counter)
                writer.add_scalar('Val_trian/base_loss_lc', base_loss_lc_data.val(), counter)
                writer.add_scalar('Val_trian/Loss', loss_data.val(), counter)
                writer.add_scalar('Val_trian/acc_gl', prec_acc_gl.val_acc(), counter)
                writer.add_scalar('Val_trian/acc_lc', prec_acc_lc.val_acc(), counter)
                writer.add_scalar('Val_trian/Acc', prec_acc.val_acc(), counter)
            counter += 1
        
    # Final grad update
    nn.utils.clip_grad_norm_(model.parameters(), 10.)
    optimizer.step()
    optimizer.zero_grad()

    # Display and save avg result
    with SummaryWriter(args.summary_dir) as writer: 
        # base loss & base acc
        writer.add_scalar('Avg_trian/base_loss_gl', base_loss_gl_data.avg(), epoch)
        writer.add_scalar('Avg_trian/base_loss_lc', base_loss_lc_data.avg(), epoch)
        writer.add_scalar('Avg_trian/Loss', loss_data.avg(), epoch)
        writer.add_scalar('Avg_trian/Acc_gl', prec_acc_gl.avg_acc(), epoch)
        writer.add_scalar('Avg_trian/Acc_lc', prec_acc_lc.avg_acc(), epoch)
        writer.add_scalar('Avg_trian/Acc', prec_acc.avg_acc(), epoch)
        # each label acc
        for i, lab_acc in enumerate(prec_acc.label_acc()):
            writer.add_scalar('Avg_trian_label/acc'+str(i), lab_acc, epoch)
        heatmap = util.heatmap(prec_acc.confus_matrix(), args.dataset)
        writer.add_figure(tag='Train Confusion Matrix',
                          figure=heatmap, global_step=epoch)
        # var loss of attention
        for j, layer_num in enumerate(args.trans_layer):
            writer.add_scalar('Avg_trian/var_loss_'+layer_num, var_loss_data.avg()[j], epoch)
        if args.pool_type in ['avg', 'vit']:
            writer.add_scalar('Avg_trian/var_loss_4', var_loss_data.avg()[-1], epoch)
            if args.local_start > 0:
                writer.add_scalar('Avg_trian/var_loss_lc', var_loss_lc_data.avg(), epoch)
    print('{:_<75}'.format(''))
    print('\033[1;32m'
        'Train [{epoch:0>3}]  Loss:{loss:0<13.11f}  Acc:{acc:0<8.6f}  Time:{time:.2f}\n'
        '[global]  base loss:{gl_base:0<13.11f}  var loss:{gl_var:0<13.11f}  acc:{gl_acc:0<8.6f}\n'
        '[local]   base loss:{lc_base:0<13.11f}  var loss:{lc_var:0<13.11f}  acc:{lc_acc:0<8.6f}'
        '\033[0m'.format(
        epoch=epoch, loss=loss_data.avg(), gl_base=base_loss_gl_data.avg(), gl_var=var_loss_data.avg().sum(),
        lc_base=base_loss_lc_all_data.avg().mean() if args.local_start > 0 else 0, lc_var=var_loss_lc_data.avg(),
        acc=prec_acc.avg_acc(), gl_acc=prec_acc_gl.avg_acc(), lc_acc=prec_acc_lc.avg_acc(), time=time.time()-start))
    return counter


# Validate
def validate(args, val_loader, model, criterion, epoch):
    with torch.no_grad():
        start = time.time()
        # total
        prec_acc = util.MatrixMeter(args.num_classes)
        loss_data = util.AverageMeter()
        # global
        prec_acc_gl = util.MatrixMeter(args.num_classes)
        base_loss_gl_data = util.AverageMeter()
        var_loss_data = util.AverageMeter()
        # local
        prec_acc_lc = util.MatrixMeter(args.num_classes)
        patch_acc = util.MatrixMeter(args.num_classes, args.patch_num[0]*args.patch_num[1])
        base_loss_lc_all_data = util.AverageMeter()
        base_loss_lc_data = util.AverageMeter()
        var_loss_lc_data = util.AverageMeter()
        

        # switch to evaluate mode
        model.eval()

        for img, label in val_loader:
            # Forward propagation
            input_var = torch.autograd.Variable(img.cuda()) # [B, 3, 224, 224] ??????????????????
            target_var = torch.autograd.Variable(label.cuda()) # [B,1] ??????????????????
            total_score, score_gl, score_lc, score_lc_all, attention_gl, attention_lc = model(input_var)

            # Loss
            ## global
            ### base loss
            base_loss_gl = criterion['base'](score_gl, target_var) # cross entropy?????????
            loss = base_loss_gl
            ### variance loss
            if args.var_loss:
                var_loss_gl = criterion['var'](attention_gl)
                loss += args.var_rate * var_loss_gl.sum()

            ## local
            if args.local_start > 0:
                ### base loss
                base_loss_lc_all = []
                for scl in score_lc_all.transpose(0,1):
                    base_loss_lc_all.append(criterion['base'](scl, target_var)) # ??????local branch???cross entropy?????????
                base_loss_lc_all = torch.stack(base_loss_lc_all, dim=0) # [P] ??????local ???????????????base loss
                ### (avg) base loss 
                base_loss_lc = criterion['base'](score_lc, target_var) # ???????????????cross entropy??????
                loss += base_loss_lc
                ### var loss
                if args.var_loss:
                    var_loss_lc = criterion['var'](attention_lc)
                    loss = loss + args.var_rate_lc * var_loss_lc
                ### default
            else:
                base_loss_lc_all = torch.Tensor([0]).cuda()
                base_loss_lc = torch.Tensor([0]).cuda()
                var_loss_lc = torch.Tensor([0]).cuda()

            # Update data
            ## total
            prec_acc.update(total_score.cpu(), label.cpu())
            loss_data.update(loss.item())
            ## global
            prec_acc_gl.update(score_gl.cpu(), label.cpu())
            base_loss_gl_data.update(base_loss_gl.item())
            var_loss_data.update(var_loss_gl.detach().cpu().numpy() * args.var_rate)
            ## local
            if args.local_start > 0:
                prec_acc_lc.update(score_lc.cpu(), label.cpu())
                patch_acc.update(score_lc_all.cpu(), label.cpu())
                base_loss_lc_all_data.update(base_loss_lc_all.cpu().numpy())
                base_loss_lc_data.update(base_loss_lc.item())
                var_loss_lc_data.update(var_loss_lc.item() * args.var_rate_lc)

        # Display and save avg result 
        with SummaryWriter(args.summary_dir) as writer: 
            writer.add_scalar('Avg_test/base_loss_gl', base_loss_gl_data.avg(), epoch)
            writer.add_scalar('Avg_test/base_loss_lc', base_loss_lc_data.avg(), epoch)
            writer.add_scalar('Avg_test/Loss', loss_data.avg(), epoch)
            writer.add_scalar('Avg_test/Acc_gl', prec_acc_gl.avg_acc(), epoch)
            writer.add_scalar('Avg_test/Acc_lc', prec_acc_lc.avg_acc(), epoch)
            writer.add_scalar('Avg_test/Acc', prec_acc.avg_acc(), epoch)
            # each label acc
            for i, lab_acc in enumerate(prec_acc.label_acc()):
                writer.add_scalar('Avg_test_label/acc'+str(i), lab_acc, epoch)
            heatmap = util.heatmap(prec_acc.confus_matrix(), args.dataset)
            writer.add_figure(tag='Test Confusion Matrix',
                              figure=heatmap, global_step=epoch)
            # var loss of attention
            for j, layer_num in enumerate(args.trans_layer):
                writer.add_scalar('Avg_test/var_loss_'+layer_num, var_loss_data.avg()[j], epoch)
            if args.pool_type in ['avg', 'vit']:
                writer.add_scalar('Avg_test/var_loss_4', var_loss_data.avg()[-1], epoch)
                if args.local_start > 0:
                    writer.add_scalar('Avg_test/var_loss_lc', var_loss_lc_data.avg(), epoch)
            if args.local_start > 0:
                # each patch loss
                for i, lc_loss in enumerate(base_loss_lc_all_data.avg()):
                    writer.add_scalar('Local_loss_test/patch_{}'.format(i), lc_loss, epoch)
                # each patch acc
                for i, lc_acc in enumerate(patch_acc.patch_acc()):
                    writer.add_scalar('Local_acc_test/patch_{}'.format(i), lc_acc, epoch)
        print('{:_<75}'.format(''))
        print('\033[1;31m'
            'Test  [{epoch:0>3}]  Loss:{loss:0<13.11f}  Acc:{acc:0<8.6f}  Time:{time:.2f}\n'
            '[global]  base loss:{gl_base:0<13.11f}  var loss:{gl_var:0<13.11f}  acc:{gl_acc:0<8.6f}\n'
            '[local]   base loss:{lc_base:0<13.11f}  var loss:{lc_var:0<13.11f}  acc:{lc_acc:0<8.6f}'
            '\033[0m'.format(
            epoch=epoch, loss=loss_data.avg(), gl_base=base_loss_gl_data.avg(), gl_var=var_loss_data.avg().sum(),
            lc_base=base_loss_lc_all_data.avg().mean() if args.local_start > 0 else 0, lc_var=var_loss_lc_data.avg(),
            acc=prec_acc.avg_acc(), gl_acc=prec_acc_gl.avg_acc(), lc_acc=prec_acc_lc.avg_acc(), time=time.time()-start))
    return prec_acc.avg_acc()

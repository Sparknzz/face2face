from dataloaders import make_data_loader
from models import model
from models import metric
from models.loss import *
from utils import *
from common import *
import os
import torch.nn.functional as F

# os.environ['CUDA_VISIBLE_DEVICES']='0'

def run_train():

    iter_accum = 1
    
    out_dir = os.getcwd()

    initial_checkpoint = None

    batch_size =  512 #8

    ## setup  -----------------------------------------------------------------------------
    for f in ['checkpoint','train']: 
        os.makedirs(out_dir +'/'+f, exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    train_loader, NUM_CLASS = make_data_loader(batch_size=batch_size)

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = model.MultiHead().cuda()
    net = nn.DataParallel(net)

    metric_1 = metric.ArcFace(512, NUM_CLASS, device_id=[0]).cuda()
    metric_2 = metric.ArcFace(512, NUM_CLASS, device_id=[0]).cuda()
    metric_3 = metric.ArcFace(512, NUM_CLASS, device_id=[0]).cuda()
    metric_4 = metric.ArcFace(512, NUM_CLASS, device_id=[0]).cuda()
    metric_5 = metric.ArcFace(512, NUM_CLASS, device_id=[0]).cuda()

    criterion = Face2Face_loss()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    if initial_checkpoint is not None:
        state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        ##for k in ['logit.weight','logit.bias']: state_dict.pop(k, None) #tramsfer sigmoid feature to softmax network
        ##net.load_state_dict(state_dict,strict=False)
        net.load_state_dict(state_dict,strict=False)
    else:
        net.module.load_pretrain(skip=['logit'])

    log.write('%s\n'%(type(net)))
    log.write('\n')

    ## optimiser ----------------------------------
    # if 0: ##freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False
    #     pass

    #net.set_mode('train',is_freeze_bn=True)
    #-----------------------------------------------

    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1, momentum=0.9, weight_decay=5e-4)
    step_size = len(train_loader) * 30 # change learning rate every 30 epoch
    schduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)


    num_iters   = 8000*1000
    iter_log    = 500
    iter_valid  = 1500
    iter_save   = [num_iters-1] + list(range(0, num_iters, len(train_loader)))

    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth', '_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']
            #optimizer.load_state_dict(checkpoint['optimizer'])
        pass

    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n'%(batch_size,iter_accum))
    log.write('                       |---------------- TRAIN/BATCH ------------- |\n')
    log.write('rate      iter   epoch |  loss   hit_accuracy    | time         \n')
    log.write('--------------------------------------------------------------------\n')
              #0.00000    0.0*   0.0  |  0.000     0            |  0 hr 00 min

    train_loss = 0.0
    batch_loss = 0.0
    train_acc = 0.0
    iter = 0
    i    = 0

    start = timer()

    top1 = AverageMeter()

    while  iter<num_iters:
        sum_train_loss = 0.0
        sum_train_acc = 0.0
        sum = 0

        optimizer.zero_grad()
        for t, (input, truth_label) in enumerate(train_loader):

            iter  = i + start_iter
            epoch = (iter-start_iter)/len(train_loader) + start_epoch

            # #if 0:
            # if (iter % iter_valid==0):
            #     valid_loss = do_valid(net, valid_loader, out_dir) #
            #     #pass

            if (iter % iter_log==0):
                print('\r',end='',flush=True)
                asterisk = '*' if iter in iter_save else ' '
                log.write('%0.5f  %5.1f%s %5.1f |  %5.3f   %4.2f | %s' % (\
                         rate, iter/1000, asterisk, epoch,
                         train_loss,
                         train_acc,
                         time_to_str((timer() - start),'min'))
                )
                log.write('\n')

            #if 0:
            if iter in iter_save:
                torch.save(net.module.networks[0].state_dict(),out_dir +'/checkpoint/%08d_model_0.pth'%(iter))
                torch.save(net.module.networks[1].state_dict(),out_dir +'/checkpoint/%08d_model_1.pth'%(iter))
                torch.save(net.module.networks[2].state_dict(),out_dir +'/checkpoint/%08d_model_2.pth'%(iter))
                torch.save(net.module.networks[3].state_dict(),out_dir +'/checkpoint/%08d_model_3.pth'%(iter))

                torch.save({
                    #'optimizer': optimizer.state_dict(),
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                pass

            # learning rate schduler -------------
            rate = get_learning_rate(optimizer)

            net.train()

            # split input images to for parts
            chunk_dim = 2
            top_split = torch.chunk(input, chunk_dim, dim=2)[0]
            left_right_split = torch.chunk(input, chunk_dim, dim=3)

            top_split = F.interpolate(top_split, size=112) 
            left_split = F.interpolate(left_right_split[0], size=112) 
            right_split = F.interpolate(left_right_split[1], size=112) 

            input = input.cuda()
            truth_label = truth_label.cuda()

            f_feat, l_feat, r_feat, t_feat, fusion_feat = net((input, top_split.cuda(), left_split.cuda(), right_split.cuda()))  #net(input)

            f_logits = metric_1(f_feat, truth_label) # batch * 512
            l_logits = metric_2(l_feat, truth_label) # batch * 512
            r_logits = metric_3(r_feat, truth_label) # batch * 512
            t_logits = metric_4(t_feat, truth_label) # batch * 512
            fusion_logits = metric_5(fusion_feat, truth_label) # batch * 512

            # cal acc
            train_acc = cal_accuracy(fusion_logits.data, truth_label, topk=(1, ))
            loss = criterion((f_logits, l_logits, r_logits, t_logits, fusion_logits), truth_label)

            (loss/iter_accum).backward()
            if (iter % iter_accum)==0:
                optimizer.step()
                optimizer.zero_grad()
                schduler.step()

            # print statistics  ------------
            batch_loss = loss.item()
            sum_train_loss += batch_loss * batch_size            
            sum += batch_size

            print('\r',end='',flush=True)
            asterisk = ' '
            print('%0.5f  %5.1f%s %5.1f |   %5.3f         %4.2f  | %s' % (\
                         rate, iter/1000, asterisk, epoch,
                         batch_loss,
                         train_acc,
                         time_to_str((timer() - start),'min')), end='',flush=True)
            i=i+1

        pass  #-- end of one data loader --
    pass #-- end of all iterations --

    log.write('\n')

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()

import os
import time
import copy
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import util_args


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # parse args
    parser = argparse.ArgumentParser(description="train LRDG",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    util_args.add_args(parser)
    parser.set_defaults(
        # record
        record_name='train_lrdg',
        saver=True,
        savedir='./checkpoints',
        logdir='./log',
        # data
        domain='pacs',
        src='art,photo,cartoon',
        trg='sketch',
        datadir='./input',
        train_trans='train',
        test_trans='test',
        # model
        network='resnet18',
        # to heavily remove the learned domain-specific features,
        # set larger lambda2 and smaller lambda3, e.g., lambda2=10, lambda3=0.0001,
        # which would show a better performance on Sketch (as target domain).
        lambda2=0.01,
        lambda3=0.1,
        resume='##',
        # optimization
        batch_size=10,
        num_epochs=1200,
        lr=.001,
        mom=.9,
        wd=1e-4,
        seed=9,
        gpus='0',
        num_workers=0,
        )

    args = parser.parse_args()

    import utils

    args.record_prefix = '{}_{}'.format(args.trg, args.record_name)

    # logging and log dir. must before tensorboard
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    log_file = os.path.join(args.logdir, '{}.log'.format(args.record_prefix))
    logging.basicConfig(filename=log_file,
                        filemode='w', level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    # tensorboard
    log_dir = os.path.join(args.logdir, args.record_prefix)
    utils.delete_existing(log_dir)
    writer = SummaryWriter(log_dir)

    # save dir
    # if not os.path.isdir(args.savedir):
    #     os.mkdir(args.savedir)
    # if args.saver:
    #     model_dir = os.path.join(args.savedir, args.record_prefix)
    #     utils.delete_existing(model_dir)
    #     os.makedirs(model_dir)

    from models import metrics
    from models import losses
    from models import optimizers
    from models.datasets import datasets
    from models.nns import resnet
    from models.nns import unet

    # --------------------------------------------------------------------------
    # print message to file
    logging.info('start with arguments %s', args)

    # set random state
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # devices for training
    devs = utils.set_devs()
    device = torch.device("cuda:{}".format(devs[0]) if devs is not None else "cpu")

    # --------------------------------------------------------------------------
    # data
    domain_datasets = datasets.get_data(args.domain)
    args.src_list = [e for e in args.src.split(',') if e != '']

    for e in args.src_list:
        domain_datasets.datasets_kfold[e]['train'].transform = datasets.data_transforms[args.train_trans]
        domain_datasets.datasets_kfold[e]['val'].transform = datasets.data_transforms[args.test_trans]
    domain_datasets.datasets[args.trg].transform = datasets.data_transforms[args.test_trans]

    src_datasets = [domain_datasets.datasets_kfold[e]['train'] for e in args.src_list]
    src_val_datasets = [domain_datasets.datasets_kfold[e]['val'] for e in args.src_list]
    trg_dataset = domain_datasets.datasets[args.trg]

    # dataloader
    src_dataloaders = [torch.utils.data.DataLoader(e, batch_size=args.batch_size, shuffle=True, sampler=None, batch_sampler=None, num_workers=args.num_workers, pin_memory=True, drop_last=True, timeout=0, worker_init_fn=None) for e in src_datasets]

    src_val_dataloaders = [torch.utils.data.DataLoader(e, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=args.num_workers, pin_memory=True, drop_last=False, timeout=0, worker_init_fn=None) for e in src_val_datasets]

    trg_dataloader = torch.utils.data.DataLoader(trg_dataset, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=args.num_workers, pin_memory=True, drop_last=False, timeout=0, worker_init_fn=None)

    args.num_classes = domain_datasets.num_class

    # --------------------------------------------------------------------------
    # cnn
    if args.network == 'resnet18':
        net = resnet.resnet18(pretrained=True)
    elif args.network == 'resnet50':
        net = resnet.resnet50(pretrained=True)
    net.fc = torch.nn.Linear(net.fc.in_features, args.num_classes, bias=False)

    # domain-specific classifiers
    F_nets = [copy.deepcopy(net) for e in args.src_list]
    # domain-invaraint classifier
    F_cla_net = copy.deepcopy(net)
    # encoder-decoder network
    mirror_net = unet.UNet(n_channels=3, n_classes=3, bilinear=True)

    # saved state for domain-specific classifiers
    resumes = [e for e in args.resume.split('#') if e != '']
    for anet, aresume in zip(F_nets, resumes):
        saved_state = torch.load(aresume)
        utils.load_state_dict(anet, saved_state)

    F_nets = [anet.to(device) for anet in F_nets]
    F_cla_net.to(device)
    mirror_net.to(device)

    F_cla_net_params = F_cla_net.parameters()
    mirror_net_params = mirror_net.parameters()

    # freeze the parameters in domain-specific classifiers
    for anet in F_nets:
        named_params = anet.named_parameters()
        for name, param in named_params:
            param.requires_grad = False

    # --------------------------------------------------------------------------
    # loss functions
    # classification loss
    ce_loss = nn.CrossEntropyLoss()
    # uncertainty loss
    enm_loss = losses.EntropyMaximization()
    # image reconstruciton loss
    mse_loss = nn.MSELoss(reduction='elementwise_mean')

    # learning rate and optimizer
    init_lr = args.lr
    optim_params = [{'params': F_cla_net.parameters(), 'fix_lr': False},
                    {'params': mirror_net.parameters(), 'fix_lr': False}]
    optimizer = optim.SGD(optim_params, lr=init_lr, momentum=args.mom,
                          dampening=0, weight_decay=args.wd, nesterov=True)

    # --------------------------------------------------------------------------
    # training
    print('training LRDG')
    logging.info('training LRDG')

    for epoch in range(args.num_epochs):

        epoch_time = utils.AverageMeter()
        eval_time = utils.AverageMeter()
        update_time = utils.AverageMeter()
        losses_bce = utils.AverageMeter()
        losses_ce = utils.AverageMeter()
        losses_mse = utils.AverageMeter()
        losses_enm = utils.AverageMeter()

        end = time.time()

        # evaluation
        for anet, loader, src_name in zip(F_nets, src_dataloaders, args.src_list):
            cla_acc_src, acc_src = metrics.compute_accuracy_class(loader, nn.Sequential(mirror_net, anet), device)
            # cla_acc_dic_trg = {}
            # for e, val in zip(domain_datasets.category_list, cla_acc_trg):
            #     cla_acc_dic_trg[e] = val
            # writer.add_scalars('eval/class_acc_trg', cla_acc_dic_trg, epoch+1)
            writer.add_scalar('eval/acc_Fds_src_{}'.format(src_name), acc_src, epoch+1)
            print("Epoch [{}]. source {} acc Fds: {}".format(epoch, src_name, acc_src))
            logging.info("Epoch [{}]. source {} acc Fds: {}".format(epoch, src_name, acc_src))

        for anet, loader, src_name in zip(F_nets, src_val_dataloaders, args.src_list):
            cla_acc_src, acc_src = metrics.compute_accuracy_class(loader, nn.Sequential(mirror_net, anet), device)
            # cla_acc_dic_trg = {}
            # for e, val in zip(domain_datasets.category_list, cla_acc_trg):
            #     cla_acc_dic_trg[e] = val
            # writer.add_scalars('eval/class_acc_trg', cla_acc_dic_trg, epoch+1)
            writer.add_scalar('eval/acc_Fds_src_val_{}'.format(src_name), acc_src, epoch+1)
            print("Epoch [{}]. source {} val acc Fds: {}".format(epoch, src_name, acc_src))
            logging.info("Epoch [{}]. source {} val acc Fds: {}".format(epoch, src_name, acc_src))

        for loader, src_name in zip(src_dataloaders, args.src_list):
            cla_acc_src, acc_src = metrics.compute_accuracy_class(loader, nn.Sequential(mirror_net, F_cla_net), device)
            # cla_acc_dic_trg = {}
            # for e, val in zip(domain_datasets.category_list, cla_acc_trg):
            #     cla_acc_dic_trg[e] = val
            # writer.add_scalars('eval/class_acc_trg', cla_acc_dic_trg, epoch+1)
            writer.add_scalar('eval/acc_F_src_{}'.format(src_name), acc_src, epoch+1)
            print("Epoch [{}]. source {} acc F: {}".format(epoch, src_name, acc_src))
            logging.info("Epoch [{}]. source {} acc F: {}".format(epoch, src_name, acc_src))

        for loader, src_name in zip(src_val_dataloaders, args.src_list):
            cla_acc_src, acc_src = metrics.compute_accuracy_class(loader, nn.Sequential(mirror_net, F_cla_net), device)
            # cla_acc_dic_trg = {}
            # for e, val in zip(domain_datasets.category_list, cla_acc_trg):
            #     cla_acc_dic_trg[e] = val
            # writer.add_scalars('eval/class_acc_trg', cla_acc_dic_trg, epoch+1)
            writer.add_scalar('eval/acc_F_src_val_{}'.format(src_name), acc_src, epoch+1)
            print("Epoch [{}]. source {} val acc F: {}".format(epoch, src_name, acc_src))
            logging.info("Epoch [{}]. source {} val acc F: {}".format(epoch, src_name, acc_src))

        cla_acc_trg, acc_trg = metrics.compute_accuracy_class(trg_dataloader, nn.Sequential(mirror_net, F_cla_net), device)
        # cla_acc_dic_trg = {}
        # for e, val in zip(domain_datasets.category_list, cla_acc_trg):
        #     cla_acc_dic_trg[e] = val
        # writer.add_scalars('eval/class_acc_trg', cla_acc_dic_trg, epoch+1)
        writer.add_scalar('eval/acc_trg', acc_trg, epoch+1)
        print("Epoch [{}]. target acc: {}".format(epoch, acc_trg))
        logging.info("Epoch [{}]. target acc: {}".format(epoch, acc_trg))

        eval_time.update(time.time() - end)
        print('Evaluation time: {}sec'.format(eval_time.val))
        logging.info('Evaluation time: {}sec'.format(eval_time.val))

        # save model
        # if (epoch+1) % 100 == 0:
        #     torch.save({'F_cla_net': F_cla_net.state_dict(),
        #                 'mirror_net': mirror_net.state_dict()},
        #                os.path.join(model_dir, '{:05d}.pth'.format(epoch+1)))
        #     print('model saved.')
        #     logging.info('model saved.')

        # CNN training mode
        F_cla_net.train()
        mirror_net.train()

        # adjust learning rate
        lr_cur = optimizers.adjust_learning_rate_clr(optimizer, init_lr, epoch)
        print("lr {}".format(lr_cur))
        logging.info("lr {}".format(lr_cur))
        writer.add_scalar('training/lr', lr_cur, epoch+1)

        begin = time.time()

        for batch_idx, batches in enumerate(zip(*src_dataloaders)):

            # zero the parameter gradients
            optimizer.zero_grad()

            imgs_src = [abatch['image'] for abatch in batches]
            lbls_src = [abatch['label'] for abatch in batches]

            len_src = imgs_src[0].size(0) * len(imgs_src)

            imgs_src = [imgs.to(device) for imgs in imgs_src]
            lbls_src = [lbls.to(device) for lbls in lbls_src]

            with torch.set_grad_enabled(True):
                recs_src = [mirror_net(imgs) for imgs in imgs_src]
                advs_src = [anet(recs) for anet, recs in zip(F_nets, recs_src)]

                loss_mse = [mse_loss(recs, imgs) for recs, imgs in zip(recs_src, imgs_src)]
                loss_enm = [enm_loss(advs, advs) for advs in advs_src]
                loss_cla = [ce_loss(F_cla_net(recs), lbls) for recs, lbls in zip(recs_src, lbls_src)]

                loss_enm_sum = torch.sum(torch.stack(loss_enm))
                loss_mse_sum = torch.sum(torch.stack(loss_mse))
                loss_cla_sum = torch.sum(torch.stack(loss_cla))

                loss = args.lambda2 * loss_enm_sum + args.lambda3 * loss_mse_sum + loss_cla_sum
                loss.backward()
                optimizer.step()

            losses_ce.update(loss_cla_sum.item(), len_src)
            losses_mse.update(loss_mse_sum.item(), len_src)
            losses_enm.update(loss_enm_sum.item(), len_src)

            # free cuda memory
            del recs_src
            del advs_src
            torch.cuda.empty_cache()

        epoch_time.update(time.time() - begin)
        print('Epoch training time: {}sec'.format(epoch_time.val))
        logging.info('Epoch training time: {}sec'.format(epoch_time.val))

        print("Epoch[{}]. average classification loss: {}".format(epoch, losses_ce.avg))
        logging.info("Epoch[{}]. average classification loss: {}".format(epoch, losses_ce.avg))
        writer.add_scalar('loss/cla', losses_ce.avg, epoch+1)

        print("Epoch[{}]. average uncertainty loss: {}".format(epoch, losses_enm.avg))
        logging.info("Epoch[{}]. average uncertainty loss: {}".format(epoch, losses_enm.avg))
        writer.add_scalar('loss/unc', losses_enm.avg, epoch+1)

        print("Epoch[{}]. average reconstruction loss: {}".format(epoch, losses_mse.avg))
        logging.info("Epoch[{}]. average reconstruction loss: {}".format(epoch, losses_mse.avg))
        writer.add_scalar('loss/rec', losses_mse.avg, epoch+1)

    # --------------------------------------------------------------------------
    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()

import os
import time
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
    parser = argparse.ArgumentParser(description="train domain-specific classifiers",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    util_args.add_args(parser)
    parser.set_defaults(
        # record
        record_name='train_ds',
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
        # optimization
        batch_size=200,
        num_epochs=1000,
        lr=.001,
        mom=.9,
        wd=1e-4,
        seed=9,
        gpus='0',
        num_workers=0,
        )

    args = parser.parse_args()

    import utils

    args.record_prefix = '{}_{}_{}'.format(args.trg, args.src, args.record_name)

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
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)
    if args.saver:
        model_dir = os.path.join(args.savedir, args.record_prefix)
        utils.delete_existing(model_dir)
        os.makedirs(model_dir)

    from models import metrics
    from models import losses
    from models.datasets import datasets
    from models.nns import resnet

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

    # domain-specific classifier for the first source domain
    src_dataset = domain_datasets.datasets_kfold[args.src_list[0]]['train']
    src_dataset_adv = torch.utils.data.ConcatDataset([domain_datasets.datasets_kfold[e]['train'] for e in args.src_list[1:]])
    src_dataset_adv_list = [domain_datasets.datasets_kfold[e]['train'] for e in args.src_list[1:]]

    src_val_dataset = domain_datasets.datasets_kfold[args.src_list[0]]['val']
    src_val_dataset_adv_list = [domain_datasets.datasets_kfold[e]['val'] for e in args.src_list[1:]]

    trg_dataset = domain_datasets.datasets[args.trg]

    # dataloader
    src_dataloader = torch.utils.data.DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, batch_sampler=None, num_workers=args.num_workers, pin_memory=True, drop_last=True, timeout=0, worker_init_fn=None)

    src_dataloader_adv = torch.utils.data.DataLoader(src_dataset_adv, batch_size=args.batch_size, shuffle=True, sampler=None, batch_sampler=None, num_workers=args.num_workers, pin_memory=True, drop_last=True, timeout=0, worker_init_fn=None)

    src_dataloader_adv_list = [torch.utils.data.DataLoader(e, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=args.num_workers, pin_memory=True, drop_last=False, timeout=0, worker_init_fn=None) for e in src_dataset_adv_list]

    src_val_dataloader = torch.utils.data.DataLoader(src_val_dataset, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=args.num_workers, pin_memory=True, drop_last=False, timeout=0, worker_init_fn=None)

    src_val_dataloader_adv_list = [torch.utils.data.DataLoader(e, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=args.num_workers, pin_memory=True, drop_last=False, timeout=0, worker_init_fn=None) for e in src_val_dataset_adv_list]

    trg_dataloader = torch.utils.data.DataLoader(trg_dataset, batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=args.num_workers, pin_memory=True, drop_last=False, timeout=0, worker_init_fn=None)

    args.num_classes = domain_datasets.num_class

    # --------------------------------------------------------------------------
    # cnn
    # domain-specific classifier
    if args.network == 'resnet18':
        net = resnet.resnet18(pretrained=True)
    elif args.network == 'resnet50':
        net = resnet.resnet50(pretrained=True)
    net.fc = torch.nn.Linear(net.fc.in_features, args.num_classes, bias=False)

    net.to(device)
    # --------------------------------------------------------------------------
    # loss functions
    # classification loss
    ce_loss = nn.CrossEntropyLoss()
    # uncertainty loss
    enm_loss = losses.EntropyMaximization()

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom,
                          dampening=0, weight_decay=args.wd, nesterov=True)

    # --------------------------------------------------------------------------
    # training
    print('training the domain-specific classifier for {}'.format(args.src_list[0]))
    logging.info('training the domain-specific classifier for {}'.format(args.src_list[0]))

    for epoch in range(args.num_epochs):

        epoch_time = utils.AverageMeter()
        eval_time = utils.AverageMeter()
        update_time = utils.AverageMeter()
        losses_bce = utils.AverageMeter()
        losses_ce = utils.AverageMeter()
        losses_enm = utils.AverageMeter()

        end = time.time()

        # evaluation
        cla_acc_src, acc_src = metrics.compute_accuracy_class(src_dataloader, net, device)
        # cla_acc_dic_src = {}
        # for e, val in zip(domain_datasets.category_list, cla_acc_src):
        #     cla_acc_dic_src[e] = val
        # writer.add_scalars('eval/class_acc_src', cla_acc_dic_src, epoch+1)
        writer.add_scalar('eval/acc_src_{}'.format(args.src_list[0]), acc_src, epoch+1)
        print("Epoch [{}]. source {} acc: {}".format(epoch, args.src_list[0], acc_src))
        logging.info("Epoch [{}]. source {} acc: {}".format(epoch, args.src_list[0], acc_src))

        cla_acc_src, acc_src = metrics.compute_accuracy_class(src_val_dataloader, net, device)
        # cla_acc_dic_src = {}
        # for e, val in zip(domain_datasets.category_list, cla_acc_src):
        #     cla_acc_dic_src[e] = val
        # writer.add_scalars('eval/class_acc_src', cla_acc_dic_src, epoch+1)
        writer.add_scalar('eval/acc_src_val_{}'.format(args.src_list[0]), acc_src, epoch+1)
        print("Epoch [{}]. source {} val acc: {}".format(epoch, args.src_list[0], acc_src))
        logging.info("Epoch [{}]. source {} val acc: {}".format(epoch, args.src_list[0], acc_src))

        for loader, src_name in zip(src_dataloader_adv_list, args.src_list[1:]):
            cla_acc_src, acc_src = metrics.compute_accuracy_class(loader, net, device)
            # cla_acc_dic_trg = {}
            # for e, val in zip(domain_datasets.category_list, cla_acc_trg):
            #     cla_acc_dic_trg[e] = val
            # writer.add_scalars('eval/class_acc_trg', cla_acc_dic_trg, epoch+1)
            writer.add_scalar('eval/acc_src_{}'.format(src_name), acc_src, epoch+1)
            print("Epoch [{}]. source {} acc: {}".format(epoch, src_name, acc_src))
            logging.info("Epoch [{}]. source {} acc: {}".format(epoch, src_name, acc_src))

        for loader, src_name in zip(src_val_dataloader_adv_list, args.src_list[1:]):
            cla_acc_src, acc_src = metrics.compute_accuracy_class(loader, net, device)
            # cla_acc_dic_trg = {}
            # for e, val in zip(domain_datasets.category_list, cla_acc_trg):
            #     cla_acc_dic_trg[e] = val
            # writer.add_scalars('eval/class_acc_trg', cla_acc_dic_trg, epoch+1)
            writer.add_scalar('eval/acc_src_val_{}'.format(src_name), acc_src, epoch+1)
            print("Epoch [{}]. source {} val acc: {}".format(epoch, src_name, acc_src))
            logging.info("Epoch [{}]. source {} val acc: {}".format(epoch, src_name, acc_src))

        cla_acc_trg, acc_trg = metrics.compute_accuracy_class(trg_dataloader, net, device)
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
        if (epoch+1) % 1000 == 0:
            torch.save(net.state_dict(),
                       os.path.join(model_dir, '{:05d}.pth'.format(epoch+1)))
            print('model saved.')
            logging.info('model saved.')

        # CNN training mode
        net.eval()

        begin = time.time()

        for batch_idx, (batch_src, batch_src_adv) in enumerate(zip(src_dataloader, src_dataloader_adv)):

            # zero the parameter gradients
            optimizer.zero_grad()

            imgs_src = batch_src['image']
            lbls_src = batch_src['label']

            imgs_src_adv = batch_src_adv['image']
            lbls_src_adv = batch_src_adv['label']

            len_src = imgs_src.size(0)
            len_src_adv = imgs_src_adv.size(0)

            imgs_src = imgs_src.to(device)
            lbls_src = lbls_src.to(device)

            imgs_src_adv = imgs_src_adv.to(device)

            with torch.set_grad_enabled(True):
                preds_src = net(imgs_src)
                preds_src_adv = net(imgs_src_adv)
                loss_cla = ce_loss(preds_src, lbls_src)
                loss_enm = enm_loss(preds_src_adv, preds_src_adv)

                loss = loss_cla + loss_enm
                loss.backward()
                optimizer.step()

            losses_ce.update(loss_cla.item(), len_src)
            losses_enm.update(loss_enm.item(), len_src_adv)

            # free cuda memory
            del preds_src
            del preds_src_adv
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

    # --------------------------------------------------------------------------
    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()

from __future__ import print_function, division
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from m3fas_utils.dataloader_fusion import face_datareader, my_transforms
from m3fas_utils.metrics import get_metrics
from arch.model import Classifier
from tqdm import tqdm
import random
import torch.nn.functional as F
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description='chenqi_echofas')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--save_model_epoch', default=1, type=int)
    parser.add_argument('--disp_step', default=300, type=int)
    parser.add_argument('--warm_start_epoch', default=0, type=int)

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--batch_size_train', default=256, type=int)
    parser.add_argument('--batch_size_val', default=256, type=int)
    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--loss_weight', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=0.00001, type=float)
    parser.add_argument('--thre', default=0.5, type=float)

    parser.add_argument('--save_root', default='./Training_results/Cross_device_fusion/', type=str)
    parser.add_argument('--root_path', default='./', type=str)
    parser.add_argument('--model_name', default="2note", type=str) # Please change the phone device accordingly (note, s9, s21, xiaomi).

    parser.add_argument('--train_csv', default="./csv/Cross_device_csv/2note_train.csv", type=str) # Please change the phone device accordingly (note, s9, s21, xiaomi).
    parser.add_argument('--val_csv', default="./csv/Cross_device_csv/2note_val.csv", type=str) # Please change the phone device accordingly (note, s9, s21, xiaomi).
    parser.add_argument('--test_csv', default="./csv/Cross_device_csv/2note_test.csv", type=str) # Please change the phone device accordingly (note, s9, s21, xiaomi).
    return parser.parse_args()

def fix_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Validation(model, dataloader, args, thre, facenum):
    model.eval()
    GT = np.zeros((facenum,), np.int)
    PRED_f = np.zeros((facenum,), np.float)
    PRED_v = np.zeros((facenum,), np.float)
    PRED_a = np.zeros((facenum,), np.float)
    for num, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            faces = data['faces'].to('cuda')
            spects = data['spects'].to('cuda')
            labels = data['labels'].to('cuda')
            logits_f,  logits_v, logits_a = model(faces, spects)

            pred_score_f = torch.nn.functional.softmax(logits_f, 1)
            pred_score_v = torch.nn.functional.softmax(logits_v, 1)
            pred_score_a = torch.nn.functional.softmax(logits_a, 1)

            GT[num * args.batch_size_val: (num*args.batch_size_val+faces.size(0))]= labels.cpu().numpy()
            PRED_f[num * args.batch_size_val: (num*args.batch_size_val+faces.size(0))] = pred_score_f[:,1].cpu().numpy()
            PRED_v[num * args.batch_size_val: (num*args.batch_size_val+faces.size(0))] = pred_score_v[:,1].cpu().numpy()
            PRED_a[num * args.batch_size_val: (num*args.batch_size_val+faces.size(0))] = pred_score_a[:,1].cpu().numpy()
            
    acc_f, auc_f, hter_f, eer_f = get_metrics(PRED_f, GT, thre)        
    acc_v, auc_v, hter_v, eer_v = get_metrics(PRED_v, GT, thre)
    acc_a, auc_a, hter_a, eer_a = get_metrics(PRED_a, GT, thre)
    return auc_f, hter_f, eer_f, acc_f, auc_v, hter_v, eer_v, acc_v, auc_a, hter_a, eer_a, acc_a


def train(args, model):
    avg_train_loss_list = np.array([])
    train_dataset = face_datareader(csv_file=args.train_csv, transform=my_transforms(size=args.input_size))
    training_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    val_dataset = face_datareader(csv_file=args.val_csv, transform=my_transforms(size=args.input_size))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    test_dataset = face_datareader(csv_file=args.test_csv, transform=my_transforms(size=args.input_size))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Num_val_faces = len(pd.read_csv(args.val_csv, header=None))
    Num_test_faces = len(pd.read_csv(args.test_csv, header=None))

    # result folder
    res_folder_name = args.save_root + args.model_name
    if not os.path.exists(res_folder_name):
        os.makedirs(res_folder_name)
        os.mkdir(res_folder_name + '/ckpt/')
    else:
        print("WARNING: RESULT PATH ALREADY EXISTED -> " + res_folder_name)
    print('find models here: ', res_folder_name)
    writer = SummaryWriter(res_folder_name)
    f1 = open(res_folder_name + "/training_log.csv", 'a+')

    # training
    steps_per_epoch = len(training_dataloader)
    Best_HTER = 1.0

    for epoch in range(args.warm_start_epoch, args.epochs):

        step_loss = np.zeros(steps_per_epoch, dtype=np.float)
        step_loss_f = np.zeros(steps_per_epoch, dtype=np.float)
        step_loss_v = np.zeros(steps_per_epoch, dtype=np.float)
        step_loss_a = np.zeros(steps_per_epoch, dtype=np.float)

        # scheduler.step()
        for step, data in enumerate(tqdm(training_dataloader)):
            model.train()
            optimizer.zero_grad()
            faces = data['faces'].to('cuda')
            spects = data['spects'].to('cuda')
            labels = data['labels'].to('cuda')
            logits_f,  logits_v, logits_a = model(faces, spects)
            loss_f = F.cross_entropy(logits_f, labels)
            loss_v = F.cross_entropy(logits_v, labels)
            loss_a = F.cross_entropy(logits_a, labels)
            loss = loss_f + args.loss_weight*(loss_v + loss_a)

            step_loss[step] = loss
            step_loss_f[step] = loss_f
            step_loss_v[step] = loss_v
            step_loss_a[step] = loss_a

            loss.backward()
            optimizer.step()
            Global_step = epoch * steps_per_epoch + (step + 1)

            if Global_step % args.disp_step == 0:
                avg_loss = np.mean(step_loss[(step + 1) - args.disp_step: (step + 1)])
                avg_loss_f = np.mean(step_loss_f[(step + 1) - args.disp_step: (step + 1)])
                avg_loss_v = np.mean(step_loss_v[(step + 1) - args.disp_step: (step + 1)])
                avg_loss_a = np.mean(step_loss_a[(step + 1) - args.disp_step: (step + 1)])
                now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                step_log_msg = '[%s] Epoch: %d/%d | Global_step: %d | Avg_loss: %f | Avg_loss_f: %f | Avg_loss_v: %f | Avg_loss_a: %f ' % (now_time, epoch + 1, args.epochs, Global_step, avg_loss, avg_loss_f, avg_loss_v, avg_loss_a)
                writer.add_scalar('Loss/train', avg_loss, Global_step)
                print('\n', step_log_msg)

        if (epoch+1) % args.save_model_epoch == 0:
            now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            avg_train_loss = np.mean(step_loss[(step + 1) - args.disp_step: (step + 1)])
            avg_train_loss_list = np.append(avg_train_loss_list, avg_train_loss)
            log_msg = '[%s] Epoch: %d/%d | 1/10 average epoch loss: %f' % (now_time, epoch + 1, args.epochs, avg_train_loss)
            print('\n', log_msg)
            f1.write(log_msg)
            f1.write('\n')

            # validation
            print('Validating...')
            AUC_f, HTER_f, EER_f, ACC_f, AUC_v, HTER_v, EER_v, ACC_v, AUC_a, HTER_a, EER_a, ACC_a = Validation(model, val_dataloader, args, args.thre, Num_val_faces)
            val_msg_f = '[%s] Epoch: %d/%d | Global_step: %d | AUC_f: %f | HTER_f: %f | EER_f: %f | ACC_f: %f' % (now_time, epoch + 1, args.epochs, Global_step, AUC_f, HTER_f, EER_f, ACC_f)
            print('\n', val_msg_f)
            val_msg_v = '[%s] Epoch: %d/%d | Global_step: %d | AUC_v: %f | HTER_v: %f | EER_v: %f | ACC_v: %f' % (now_time, epoch + 1, args.epochs, Global_step, AUC_v, HTER_v, EER_v, ACC_v)
            print('\n', val_msg_v)
            val_msg_a = '[%s] Epoch: %d/%d | Global_step: %d | AUC_a: %f | HTER_a: %f | EER_a: %f | ACC_a: %f' % (now_time, epoch + 1, args.epochs, Global_step, AUC_a, HTER_a, EER_a, ACC_a)
            print('\n', val_msg_a)
            f1.write(val_msg_f)
            f1.write('\n')
            f1.write(val_msg_v)
            f1.write('\n')
            f1.write(val_msg_a)
            f1.write('\n')

            # Here we pick the last checkpoint as the final ckpt. You can also try some alternative strategies.
            #save model
            if not HTER_f > Best_HTER:
                # Best_HTER = HTER_f
                torch.save(model.state_dict(), res_folder_name + '/ckpt/' + 'best.pth')
                np.save(res_folder_name + '/avg_train_loss_list.np', avg_train_loss_list)
                cur_learning_rate = [param_group['lr'] for param_group in optimizer.param_groups]
                print('Saved model. lr %f' % cur_learning_rate[0])
                f1.write('Saved model. lr %f' % cur_learning_rate[0])
                f1.write('\n')

                print('Testing...')
                AUC_f, HTER_f, EER_f, ACC_f, AUC_v, HTER_v, EER_v, ACC_v, AUC_a, HTER_a, EER_a, ACC_a = Validation(model, test_dataloader, args, args.thre, Num_test_faces)
                test_msg_f =  '[%s] Epoch: %d/%d | Global_step: %d | AUC_f: %f | HTER_f: %f | EER_f: %f | ACC_f: %f' % (now_time, epoch + 1, args.epochs, Global_step, AUC_f, HTER_f, EER_f, ACC_f)
                print('\n', test_msg_f)
                test_msg_v = '[%s] Epoch: %d/%d | Global_step: %d | AUC_v: %f | HTER_v: %f | EER_v: %f | ACC_v: %f' % (now_time, epoch + 1, args.epochs, Global_step, AUC_v, HTER_v, EER_v, ACC_v)
                print('\n', test_msg_v)
                test_msg_a = '[%s] Epoch: %d/%d | Global_step: %d | AUC_a: %f | HTER_a: %f | EER_a: %f | ACC_a: %f' % (now_time, epoch + 1, args.epochs, Global_step, AUC_a, HTER_a, EER_a, ACC_a)
                print('\n', test_msg_a)

                f1.write(test_msg_f)
                f1.write('\n')
                f1.write(test_msg_v)
                f1.write('\n')
                f1.write(test_msg_a)
                f1.write('\n')
    f1.close()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.random_seed is not None:
        fix_seed(args.random_seed)
    model = Classifier()
    model = model.to('cuda')
    train(args, model)

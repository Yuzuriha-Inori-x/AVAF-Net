import sys

sys.path.append("../../AVAF-Net")
import torch
import random
import numpy as np
import torch.nn as nn
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from avqa_scripts.dataloader_music import AVQA_dataset, ToTensor
from avqa_scripts.model.net_dmcg import DMCG
import torch.optim as optim
import json
import ast

from configs.args import parser

def model_train(args, model, train_loader, optimizer, criterion, epoch=0):
    model.train()
    total_qa = 0
    correct_qa = 0
    for batch_idx, sample in enumerate(train_loader):
        audio_feature, patch_feature, target, question, ques_len, video_id, question_id = \
            sample['audio_feature'].to('cuda'), sample['video_feature'].to('cuda'), \
            sample['label'].to('cuda'), sample['question'].to('cuda'), sample['ques_len'].to('cuda'), sample[
                'video_id'], sample['question_id']

        optimizer.zero_grad()
        out_qa, nce_loss, fnac_loss = model(audio_feature, patch_feature, question, ques_len)

        loss_qa = criterion(out_qa, target)
        nce_loss = args.loss_nce_wei * nce_loss
        fnac_loss = args.loss_fnac_wei * fnac_loss
        loss = loss_qa + nce_loss + fnac_loss

        loss.backward()
        optimizer.step()

        pred_index, predicted = torch.max(out_qa, 1)
        correct_qa += (predicted == target).sum().item()
        total_qa += out_qa.size(0)

        if batch_idx % args.log_interval == 0:
            # print(
            #     'Train Epoch: {} [{}/{} ({:.0f}%)]\t total Loss: {:.4f}  |  CE_loss:{:.6f}  NCE-loss:{:.4f}  FNA-loss:{:.4f}'.format(
            #         epoch, batch_idx * len(audio_feature), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item(), loss_qa.item(), nce_loss.item(),
            #         fnac_loss.item()), flush=True)
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t total Loss: {:.4f}'.format(
                    epoch, batch_idx * len(audio_feature), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()), flush=True)

    return correct_qa, total_qa, 100 * correct_qa / total_qa

def model_eval(model, val_loader):
    model.eval()
    total_qa = 0
    correct_qa = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio_feature, patch_feature, target, question, ques_len, video_id, question_id = \
                sample['audio_feature'].to('cuda'), sample['video_feature'].to('cuda'), \
                sample['label'].to('cuda'), sample['question'].to('cuda'), sample['ques_len'].to('cuda'), sample[
                    'video_id'], sample['question_id']

            preds_qa, _, _ = model(audio_feature, patch_feature, question, ques_len)
            _, predicted = torch.max(preds_qa, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()

    print('Accuracy val_set qa: %.2f %%' % (100 * correct_qa / total_qa), flush=True)

    return 100 * correct_qa / total_qa


def model_test(model, test_loader, test_json_file):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open(test_json_file, 'r'))

    # useing index of question
    questionid_to_samples = {}
    for sample in samples:
        ques_id = sample['question_id']
        if ques_id not in questionid_to_samples.keys():
            questionid_to_samples[ques_id] = sample
        else:
            print("question_id_duplicated:", ques_id)

    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):

            audio_feature, patch_feature, target, question, ques_len, video_id, question_id = \
                sample['audio_feature'].to('cuda'), sample['video_feature'].to('cuda'), \
                sample['label'].to('cuda'), sample['question'].to('cuda'), sample['ques_len'].to('cuda'), sample[
                    'video_id'], sample['question_id']

            preds_qa, _, _ = model(audio_feature, patch_feature, question, ques_len)

            preds = preds_qa
            _, predicted = torch.max(preds.data, 1)
            total += preds.size(0)
            correct += (predicted == target).sum().item()

            question_id = question_id.numpy().tolist()

            for index, ques_id in enumerate(question_id):
                x = questionid_to_samples[ques_id]
                type = ast.literal_eval(x['type'])

                if type[0] == 'Audio':
                    if type[1] == 'Counting':
                        A_count.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Comparative':
                        A_cmp.append((predicted[index] == target[index]).sum().item())
                elif type[0] == 'Visual':
                    if type[1] == 'Counting':
                        V_count.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Location':
                        V_loc.append((predicted[index] == target[index]).sum().item())
                elif type[0] == 'Audio-Visual':
                    if type[1] == 'Existential':
                        AV_ext.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Counting':
                        AV_count.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Location':
                        AV_loc.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Comparative':
                        AV_cmp.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Temporal':
                        AV_temp.append((predicted[index] == target[index]).sum().item())

    print('Audio Counting Accuracy: %.2f %%' % (
                100 * sum(A_count) / len(A_count)))
    print('Audio Cmp Accuracy: %.2f %%' % (
                100 * sum(A_cmp) / len(A_cmp)))
    print('Audio Accuracy: %.2f %%' % (
                100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    print('Visual Counting Accuracy: %.2f %%' % (
                100 * sum(V_count) / len(V_count)))
    print('Visual Loc Accuracy: %.2f %%' % (
                100 * sum(V_loc) / len(V_loc)))
    print('Visual Accuracy: %.2f %%' % (
                100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    print('AV Ext Accuracy: %.2f %%' % (
                100 * sum(AV_ext) / len(AV_ext)))
    print('AV Loc Accuracy: %.2f %%' % (
                100 * sum(AV_loc) / len(AV_loc)))
    print('AV counting Accuracy: %.2f %%' % (
                100 * sum(AV_count) / len(AV_count)))
    print('AV Cmp Accuracy: %.2f %%' % (
                100 * sum(AV_cmp) / len(AV_cmp)))
    print('AV Temporal Accuracy: %.2f %%' % (
                100 * sum(AV_temp) / len(AV_temp)))
    print('AV Accuracy: %.2f %%' % (
                100 * (sum(AV_count) + sum(AV_loc) + sum(AV_ext) + sum(AV_temp)
                       + sum(AV_cmp)) / (len(AV_count) + len(AV_loc) + len(AV_ext) + len(AV_temp) + len(AV_cmp))))

    print('Overall Accuracy: %.2f %%' % (
                100 * correct / total))

    return 100 * correct / total

def main():
    args = parser.parse_args()
    print(format("main.py path", '<25'), Path(__file__).resolve())

    for arg in vars(args):
        print(format(arg, '<25'), format(str(getattr(args, arg)), '<'))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.model == 'DMCG-QA':
        model = DMCG(
            audio_input_dim=args.audio_input_dim,
            patch_input_dim=args.patch_input_dim,
            hidden_size=args.hidden_size,
            answer_vocab_size=args.answer_vocab_size,
            dropout_p1=args.dropout_p1,
            dropout_p2=args.dropout_p2,
            sa_encoder_layers_num=args.sa_encoder_layers_num,
            sa_nhead=args.sa_nhead,
            sa_d_model=args.sa_d_model,
            sa_dim_feedforward=args.sa_dim_feedforward,
            cx_encoder_layers_num=args.cx_encoder_layers_num,
            cx_nhead=args.cx_nhead,
            cx_d_model=args.cx_d_model,
            cx_dim_feedforward=args.cx_dim_feedforward,
            frame_len=args.frame_len,
            thread=args.thread
        )
        # model = nn.DataParallel(model)
        model = model.to('cuda')
    else:
        raise ('not recognized')

    if args.mode == 'train':
        train_dataset = AVQA_dataset(label=args.label_train, audio_dir=args.audio_dir, clip_vit_b32_dir=args.video_dir,
                                     clip_word_dir=args.word_dir, transform=transforms.Compose([ToTensor()]),
                                     mode_flag='train')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6,
                                  pin_memory=True)

        val_dataset = AVQA_dataset(label=args.label_test, audio_dir=args.audio_dir, clip_vit_b32_dir=args.video_dir,
                                   clip_word_dir=args.word_dir, transform=transforms.Compose([ToTensor()]),
                                   mode_flag='val')
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.steplr_step, gamma=args.steplr_gamma)
        criterion = nn.CrossEntropyLoss()

        val_best = 0
        for epoch in range(1, args.epochs + 1):
            print(f"\nthe {epoch}-th learning rate is {optimizer.param_groups[0]['lr']}")
            #########################################################################################################
            # !!! train
            correct_qa, total_qa, train_acc = model_train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            print('Accuracy train_set qa: %.2f %%' % (100 * correct_qa / total_qa), flush=True)
            scheduler.step(epoch)

            #########################################################################################################
            # !!! val
            val_acc = model_eval(model, val_loader)

            if val_acc >= val_best:
                model_to_save = model.module if hasattr(model, 'module') else model
                optimizer_to_save = optimizer
                val_best = val_acc
                save_model_folder = Path(args.model_save_dir, args.checkpoint_file)
                save_epoch_model_path = Path(save_model_folder, f"model_{epoch}.pt")
                save_best_model_path = Path(save_model_folder, f"model_best.pt")
                if not os.path.isdir(save_model_folder):
                    os.mkdir(save_model_folder)

                if args.save_model_flag == 'True':
                    # epoch_save_dict = {
                    #     'epoch': epoch,
                    #     'model_state_dict': model_to_save.state_dict(),
                    #     'optimizer_state_dict': optimizer_to_save.state_dict(),
                    #     'Acc': val_best,
                    # }
                    best_save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer_to_save.state_dict(),
                        'Acc': val_best,
                    }
                    # torch.save(epoch_save_dict, save_epoch_model_path)
                    torch.save(best_save_dict, save_best_model_path)

    elif args.mode == 'test':
        test_dataset = AVQA_dataset(label=args.label_test, audio_dir=args.audio_dir, clip_vit_b32_dir=args.video_dir,
                                    clip_word_dir=args.word_dir, transform=transforms.Compose([ToTensor()]),
                                    mode_flag='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6,
                                 pin_memory=True)
        checkpoint = torch.load(os.path.join(args.model_save_dir, args.checkpoint_file, 'model_best.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        # model = nn.DataParallel(model)
        model = model.to('cuda')
        best_epoch = checkpoint['epoch']
        best_acc = checkpoint['Acc']
        print("-------- checkpoint loading successfully ----------")
        print('********************************************************')
        print('The best epoch ---------- {0}'.format(best_epoch))
        print('The best train acc ------ {0}'.format(best_acc))
        print('********************************************************')
        _ = model_test(model, test_loader, args.label_test)


if __name__ == '__main__':
    main()

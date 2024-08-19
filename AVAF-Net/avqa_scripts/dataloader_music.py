import numpy as np
import torch
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import ast
import json


class AVQA_dataset(Dataset):

    def __init__(self, label, audio_dir, clip_vit_b32_dir, clip_word_dir,
                 transform=None, mode_flag='train'):
        # for loading all words.
        train_samples = json.load(open('../DMCG-QA-main/avqa_data/music_avqa/music_avqa.json', 'r'))

        ques_vocab = ['<pad>']
        ans_vocab = []
        i = 0
        for sample in train_samples:
            i += 1
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1

            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample['anser'] not in ans_vocab:
                ans_vocab.append(sample['anser'])

        self.ques_vocab = ques_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.ans_vocab = ans_vocab
        self.ans_to_idx = {id: index for index, id in enumerate(self.ans_vocab)}

        # loading train/val/test json file.
        self.samples = json.load(open(label, 'r'))
        self.max_len = 14

        self.audio_dir = audio_dir

        self.clip_vit_b32_dir = clip_vit_b32_dir
        self.clip_word_dir = clip_word_dir

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        name = sample['video_id']

        # frame_indx = np.linspace(0, 59, num=50, dtype=int)

        # * audio vggish
        audio_feature = np.load(os.path.join(self.audio_dir, name + '.npy'))
        audio_feature = audio_feature[:60, :]
        # audio_feature = audio_feature[frame_indx, :]

        # * clip patch
        visual_CLIP_feat = np.load(os.path.join(self.clip_vit_b32_dir, name + '.npy'))
        patch_feat = visual_CLIP_feat[:60, 1:, :]  # [t, 49, 512]
        # patch_feat = visual_CLIP_feat[frame_indx, 1:, :]  # [t, 49, 512]

        # * clip question
        question_id = int(sample['question_id'])
        question_feat = np.load(os.path.join(self.clip_word_dir, str(question_id) + '.npy'))[0]

        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]
        ques_len = len(question)
        ques_len = torch.from_numpy(np.array(ques_len))  # question length.

        # * answer
        answer = sample['anser']
        label = self.ans_to_idx[answer]
        label = torch.from_numpy(np.array(label)).long()

        sample = {'audio_feature': audio_feature, 'video_feature': patch_feat,
                  'question': question_feat, 'ques_len': ques_len, 'label': label,
                  'video_id': name, 'question_id': question_id}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):

    def __call__(self, sample):
        audio_feature = sample['audio_feature']
        video_feature = sample['video_feature']
        question = sample['question']
        ques_len = sample['ques_len']
        label = sample['label']
        name = sample['video_id']
        question_id = sample['question_id']

        return {
            'audio_feature': torch.from_numpy(audio_feature),
            'video_feature': video_feature,
            'question': question,
            'ques_len': ques_len,
            'label': label,
            'video_id': name,
            'question_id': question_id
        }

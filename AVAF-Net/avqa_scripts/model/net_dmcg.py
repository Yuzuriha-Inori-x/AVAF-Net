import functorch.dim
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from avqa_scripts.model.loss import AVContrastive_loss_50 as AVContrastive_loss
from avqa_scripts.model.loss import FNAContrastive_loss as FNA_loss
from avqa_scripts.model.fusion import AttFlat
from avqa_scripts.model.transformer_encoder import SAEncoder, CXEncoder


class DMCG(nn.Module):
    def __init__(self,
                 audio_input_dim=128,
                 patch_input_dim=512,
                 hidden_size=512,
                 answer_vocab_size=42,
                 dropout_p1=0.1,
                 dropout_p2=0.1,
                 sa_encoder_layers_num=1,
                 sa_nhead=1,
                 sa_d_model=512,
                 sa_dim_feedforward=2048,
                 cx_encoder_layers_num=1,
                 cx_nhead=4,
                 cx_d_model=512,
                 cx_dim_feedforward=2048,
                 frame_len=60,
                 thread=0.0110):
        super(DMCG, self).__init__()

        self.frame_len = frame_len
        self.thread = thread
        self.hidden_size = hidden_size
        self.fc_audio = nn.Linear(audio_input_dim, hidden_size)
        self.fc_patch = nn.Linear(patch_input_dim, hidden_size)

        self.sa_a_pos_embed = nn.Embedding(frame_len, hidden_size)
        self.sa_audio_encoder = SAEncoder(d_model=sa_d_model, nhead=sa_nhead, num_encoder_layers=sa_encoder_layers_num,
                                          dim_feedforward=sa_dim_feedforward, dropout=dropout_p1)

        self.aqq_encoder = CXEncoder(d_model=cx_d_model, nhead=cx_nhead, num_encoder_layers=cx_encoder_layers_num,
                                     dim_feedforward=cx_dim_feedforward, dropout=dropout_p1)
        self.pqq_encoder = CXEncoder(d_model=cx_d_model, nhead=cx_nhead, num_encoder_layers=cx_encoder_layers_num,
                                     dim_feedforward=cx_dim_feedforward, dropout=dropout_p1)

        self.qaa_encoder = CXEncoder(d_model=cx_d_model, nhead=cx_nhead, num_encoder_layers=cx_encoder_layers_num,
                                     dim_feedforward=cx_dim_feedforward, dropout=dropout_p1)
        self.qpp_encoder = CXEncoder(d_model=cx_d_model, nhead=cx_nhead, num_encoder_layers=cx_encoder_layers_num,
                                     dim_feedforward=cx_dim_feedforward, dropout=dropout_p1)

        self.wei_from_q = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU(inplace=True)
        self.attflat = AttFlat(hidden_size, hidden_size, 1, answer_vocab_size, dropout_r=dropout_p2)

    def forward(self, audio_feature, patch_feature, question, ques_len):
        out, nce_loss, fnac_loss = self.model_block(audio_feature, patch_feature, question, ques_len)
        return out, nce_loss, fnac_loss

    def model_block(self, audio_fea, patch_feat, question_feat, ques_len):
        """
        :param audio_fea: from vggish [b, t, c=128]
        :param patch_feat:  from clip-vision-encoder [b ,t ,n=49 ,c=512]
        :param question_feat: from clip-text-encoder [b, n=77, c=512]
        :param ques_len: [b, 1]
        :return:
        """
        ######################################################################################
        # Feature preparation
        # * question
        # * word level
        # print()
        word_fea = question_feat[:, 1:, :]  # [B, 76, 512]
        q_mask = self.make_mask(word_fea, ques_len)  # [B, 14]
        word_fea = word_fea.permute(1, 0, 2)  # [76, B, 512]

        # * sentence level
        sentence_fea = question_feat[:, 0, :]  # [B, 512]
        q_temp = sentence_fea.unsqueeze(1)  # [B, 1, C]
        q_repeat = q_temp.repeat(1, self.frame_len, 1)  # [B, T, C]

        # * patch
        B, T, PATCH_NUM, _ = patch_feat.shape  # [B, T, NUM, C]
        patch_feat = patch_feat.view(B, T * PATCH_NUM, -1)  # [B, T*NUM, C]
        patch_feat = self.fc_patch(patch_feat)  # [B, T*NUM, C]
        patch_feat = patch_feat.permute(1, 0, 2)  # [T*NUM, B, C]

        # * audio
        audio_fea = self.fc_audio(audio_fea)
        audio_fea = audio_fea.permute(1, 0, 2)  # [T, B, C]
        sa_a_pos = self.sa_a_pos_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [T, B, C]
        sa_audio_fea, _, _ = self.sa_audio_encoder(audio_fea, attn_mask=None,
                                                   key_padding_mask=None,
                                                   pos_embed=sa_a_pos)

        ######################################################################################
        # Temporal_Align_loss
        fnac_loss1, fnac_loss2, fnac_loss3 = FNA_loss(patch_feat.permute(1, 0, 2), sa_audio_fea.permute(1, 0, 2))
        fnac_loss = fnac_loss1 + fnac_loss2 + fnac_loss3

        ######################################################################################
        # Audio-visual clues mining
        question_mask = self.gene_question_as_key_pad_mask(word_fea.permute(1, 0, 2), ques_len)
        question_mask = question_mask.to('cuda')

        cx_a_fea, _, _ = self.aqq_encoder(sa_audio_fea, word_fea, attn_mask=None, key_padding_mask=question_mask,
                                          q_pos_embed=None, k_pos_embed=None)  # [T, bs, 512]
        cx_p_fea, _, _ = self.pqq_encoder(patch_feat, word_fea, attn_mask=None, key_padding_mask=question_mask,
                                          q_pos_embed=None, k_pos_embed=None)  # [T*N, bs, 512]

        ######################################################################################
        # Spaital_Align_loss
        nce_loss_PQ = AVContrastive_loss(cx_p_fea.permute(1, 0, 2), q_repeat, thread=self.thread)
        nce_loss_PA = AVContrastive_loss(cx_p_fea.permute(1, 0, 2), cx_a_fea.permute(1, 0, 2), thread=self.thread)
        nce_loss = nce_loss_PQ + nce_loss_PA

        ######################################################################################
        # Audio-visual Adaptive Fusion
        cx_a_fea2, _, _ = self.qaa_encoder(word_fea, cx_a_fea, attn_mask=None, key_padding_mask=None, q_pos_embed=None,
                                           k_pos_embed=None)  # [cur_max_lenth, bs, 512]
        cx_p_fea2, _, _ = self.qpp_encoder(word_fea, cx_p_fea, attn_mask=None, key_padding_mask=None, q_pos_embed=None,
                                           k_pos_embed=None)  # [cur_max_lenth, bs, 512]
        cx_a_fea2 = cx_a_fea2.permute(1, 0, 2)  # [bs, cur_max_lenth, 512]
        cx_p_fea2 = cx_p_fea2.permute(1, 0, 2)  # [bs, cur_max_lenth, 512]

        modality_wei = self.wei_from_q(sentence_fea)
        modality_wei = self.relu(modality_wei)
        modality_wei = torch.softmax(modality_wei, dim=-1)
        cx_a_fea2 = cx_a_fea2 * modality_wei[:, 0].unsqueeze(-1).unsqueeze(-1)
        cx_p_fea2 = cx_p_fea2 * modality_wei[:, 1].unsqueeze(-1).unsqueeze(-1)
        cx_fused_fea = cx_a_fea2 + cx_p_fea2
        # print('cx_fused_fea:', cx_fused_fea.shape)

        ######################################################################################
        # answer prediction
        fusion_out = self.attflat(cx_fused_fea, q_mask)
        return fusion_out, nce_loss, fnac_loss

    def gene_question_as_key_pad_mask(self, q_fea, seq_length):
        mask = torch.ones(q_fea.shape[:2])
        for i, l in enumerate(seq_length):
            mask[i][l:] = 0
        mask = mask.to(torch.bool)
        mask = ~mask
        return mask

    def make_mask(self, seq, seq_length):
        mask = torch.ones(seq.shape[:2]).cuda()
        # print('mask:', mask.shape)

        for i, l in enumerate(seq_length):
            mask[i][l:] = 0
        mask = Variable(mask)
        mask = mask.to(torch.float)
        return mask


if __name__ == '__main__':
    dmcg = DMCG(frame_len=10)
    audio_feature = torch.randn(2, 10, 128)
    patch_feature = torch.randn(2, 10, 49, 512)
    question = torch.randn(2, 77, 512)
    ques_len = torch.randint(5, 15, (2, 1))
    out1, nce_loss, fnac_loss = dmcg(audio_feature, patch_feature, question, ques_len)

    print(out1, nce_loss, fnac_loss)

    soft_max = nn.Softmax(dim=1)
    out2 = soft_max(out1)
    print(out2, nce_loss, fnac_loss)

    pred_index, predicted = torch.max(out1, 1)
    print(pred_index, predicted)

    pred_index, predicted = torch.max(out2, 1)
    print(pred_index, predicted)


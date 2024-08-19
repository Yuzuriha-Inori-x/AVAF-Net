import torch
import torch.nn as nn
import torch.nn.functional as F


def FNAContrastive_loss(img, aud, tau=0.03, high_conf_thresh=0.6):
    aud_attn = (aud @ aud.transpose(0, 1)) / tau

    img_avg = nn.AdaptiveAvgPool2d((1, 1))(img)[:, :, 0, 0]
    img_attn = (img_avg @ img_avg.transpose(0, 1)) / tau

    B = img.shape[0]
    h, w = img.shape[2], img.shape[3]

    Slogits = torch.einsum('nchw,mc->nmhw', img, aud) / tau

    loc_map = Slogits[torch.arange(B), torch.arange(B)]
    loc_map = (loc_map - torch.amin(loc_map, (1, 2), keepdim=True)) / \
              (torch.amax(loc_map, (1, 2), keepdim=True) - torch.amin(loc_map, (1, 2), keepdim=True) + 1e-5)

    # frg_feature = img * loc_map.unsqueeze(1)
    frg_feature = img * (loc_map > high_conf_thresh).unsqueeze(1)  # foreground visual features
    frg_feature = frg_feature.flatten(-2, -1).mean(dim=-1)
    frg_attn = (frg_feature @ frg_feature.transpose(0, 1)) / tau

    logits = Slogits.flatten(-2, -1).max(dim=-1)[0]
    labels = torch.arange(B).long().to(img.device)

    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.permute(1, 0), labels)

    fnac_loss1 = F.l1_loss(torch.softmax(aud_attn, dim=1), torch.softmax(logits, dim=1))  # FNS-1
    fnac_loss2 = F.l1_loss(torch.softmax(aud_attn, dim=1), torch.softmax(frg_attn, dim=1))  # TNS
    fnac_loss3 = F.l1_loss(torch.softmax(img_attn, dim=1), torch.softmax(logits, dim=1))  # FNS-2

    return [loss, fnac_loss1, fnac_loss2, fnac_loss3], Slogits


obj_feature = torch.randn(2, 512, 7, 7)
audio_feature = torch.randn(2, 512)
[loss, fnac_loss1, fnac_loss2, fnac_loss3], Slogits = FNAContrastive_loss(obj_feature, audio_feature)
print([loss, fnac_loss1, fnac_loss2, fnac_loss3])


import torch
import torch.nn as nn


class JointsL2Loss(nn.Module):
    def __init__(self, has_ohkm=False, topk=8, thresh1=1, thresh2=0):
        super(JointsL2Loss, self).__init__()
        self.has_ohkm = has_ohkm
        self.topk = topk
        self.t1 = thresh1
        self.t2 = thresh2
        method = 'none' if self.has_ohkm else 'mean'
        self.calculate = nn.MSELoss(reduction=method)

    def forward(self, output, valid, label):
        assert output.shape == label.shape
        batch_size = output.size(0)
        keypoint_num = output.size(1)
        loss = 0

        for i in range(batch_size):
            pred = output[i].reshape(keypoint_num, -1)
            gt = label[i].reshape(keypoint_num, -1)

            # occlusion
            if not self.has_ohkm:
                weight = torch.gt(valid[i], self.t1).float()
                gt = gt * weight

            tmp_loss = self.calculate(pred, gt)

            if self.has_ohkm:
                tmp_loss = tmp_loss.mean(dim=1)
                weight = torch.gt(valid[i].squeeze(), self.t2).float()
                tmp_loss = tmp_loss * weight
                # hard sample
                topk_val, topk_id = torch.topk(tmp_loss, k=self.topk, dim=0,
                                               sorted=False)
                sample_loss = topk_val.mean(dim=0)
            else:
                sample_loss = tmp_loss

            loss = loss + sample_loss
        return loss / batch_size

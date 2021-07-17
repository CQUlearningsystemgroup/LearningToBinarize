import torch
from torch.nn import functional as F
from torch.nn.modules import loss


class DistributionLoss(loss._Loss):

    def forward(self, model_output, real_output):

        self.size_average = True

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        if real_output.requires_grad:
            raise ValueError("real network output should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        real_output_soft = F.softmax(real_output, dim=1)
        del model_output, real_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        if self.size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        # model_output_log_prob = model_output_log_prob.squeeze(2)
        return cross_entropy_loss

class DistillationLoss(torch.nn.Module):

    def __init__(self, alpha=0.9):
        super(DistillationLoss, self).__init__()
        self.criterion1 = torch.nn.CrossEntropyLoss()
        self.criterion2 = DistributionLoss()
        self.alpha = alpha

    def forward(self, stu_model_output, tea_model_output, target):
        loss1 = self.criterion1(stu_model_output, target)
        loss2 = self.criterion2(stu_model_output, tea_model_output)

        loss = self.alpha * loss2 + (1. - self.alpha) * loss1

        return loss, loss1


class KdLoss(torch.nn.Module):
    def __init__(self, alpha=0.9, T = 5):
        super(KdLoss, self).__init__()
        self.alpha = alpha
        self.T= T
        self.criterion = torch.nn.KLDivLoss()

    def forward(self, outputs, teacher_outputs, labels):

        alpha = self.alpha
        T = self.T
        KD_loss = self.criterion(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs /T, dim=1)) * (alpha * T * T) + \
            F.cross_entropy(outputs, labels) * (1. - alpha)

        return KD_loss
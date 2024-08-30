import torch.nn.functional as F
import torch.nn as nn


def loss_kd(outputs, labels, teacher_outputs, temparature, alpha):
   KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/temparature, 
                                          dim=1),
                                          F.softmax(teacher_outputs/temparature,dim=1))*(alpha * temparature * temparature)+F.cross_entropy(outputs, labels) * (1. -alpha)
   return KD_loss

def get_outputs(model, dataloader):
   '''
   Used to get the output of the teacher network
   '''
   outputs = []
   for inputs, labels in dataloader:
      inputs_batch, labels_batch = inputs.cuda(), labels.cuda()
      output_batch = model(inputs_batch).data.cpu().numpy()
      outputs.append(output_batch)
   return outputs
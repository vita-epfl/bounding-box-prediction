import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# Variation of the PV_LSTM network with position decoder
class PV_LSTM(nn.Module):
    def __init__(self, args):
        self.size = 6
        
        super(PV_LSTM, self).__init__()
        
        self.speed_encoder = nn.LSTM(input_size=self.size, hidden_size=args.hidden_size)
        self.pos_encoder   = nn.LSTM(input_size=self.size, hidden_size=args.hidden_size)
        
        self.pos_embedding = nn.Sequential(nn.Linear(in_features=args.hidden_size, out_features=self.size),
                                           nn.ReLU())
        
        self.speed_decoder    = nn.LSTMCell(input_size=self.size, hidden_size=args.hidden_size)
        self.pos_decoder    = nn.LSTMCell(input_size=self.size, hidden_size=args.hidden_size)
        self.fc_speed    = nn.Linear(in_features=args.hidden_size, out_features=self.size)
              
        self.args = args
        
    def forward(self, speed=None, pos=None, average=False):

        _, (hsp, csp) = self.speed_encoder(speed.permute(1,0,2))
        hsp = hsp.squeeze(0)
        csp = csp.squeeze(0)
        
        _, (hpo, cpo) = self.pos_encoder(pos.permute(1,0,2))
        hpo = hpo.squeeze(0)
        cpo = cpo.squeeze(0)
        
        outputs = []
        
        if 'bounding_box' in self.args.task:
            speed_outputs    = torch.tensor([], device=self.args.device)
            pos_outputs    = torch.tensor([], device=self.args.device)
            in_sp = speed[:,-1,:]
            in_pos = pos[:,-1,:]
            
            hds = hpo + hsp
            cds = cpo + csp
            for i in range(self.args.output//self.args.skip):
                hds, cds         = self.speed_decoder(in_sp, (hds, cds))
                speed_output     = self.fc_speed(hds)
                speed_outputs    = torch.cat((speed_outputs, speed_output.unsqueeze(1)), dim = 1)
                
                hpo, cpo         = self.pos_decoder(in_pos, (hpo, cpo))
                pos_output       = self.fc_speed(hpo)
                pos_outputs      = torch.cat((pos_outputs, pos_output.unsqueeze(1)), dim = 1)
                
            outputs.append(speed_outputs)
            outputs.append(pos_outputs)
            
        #return tuple(outputs)
        return outputs
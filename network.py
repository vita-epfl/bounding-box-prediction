import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

class PV_LSTM(nn.Module):
    def __init__(self, args):
        super(PV_LSTM, self).__init__()
        
        self.speed_encoder = nn.LSTM(input_size=4, hidden_size=args.hidden_size)
        self.pos_encoder   = nn.LSTM(input_size=4, hidden_size=args.hidden_size)
        
        self.pos_embedding = nn.Sequential(nn.Linear(in_features=args.hidden_size, out_features=4),
                                           nn.ReLU())
        
        self.speed_decoder    = nn.LSTMCell(input_size=4, hidden_size=args.hidden_size)
        self.crossing_decoder = nn.LSTMCell(input_size=4, hidden_size=args.hidden_size)
        
        self.fc_speed    = nn.Linear(in_features=args.hidden_size, out_features=4)
        self.fc_crossing = nn.Sequential(nn.Linear(in_features=args.hidden_size, out_features=2),
                                         nn.ReLU())
        
        self.hardtanh = nn.Hardtanh(min_val=-1*args.hardtanh_limit, max_val=args.hardtanh_limit)
        self.softmax = nn.Softmax(dim=1)
        
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
            in_sp = speed[:,-1,:]
            
            hds = hpo + hsp
            cds = cpo + csp
            for i in range(self.args.output//self.args.skip):
                hds, cds         = self.speed_decoder(in_sp, (hds, cds))
                speed_output     = self.hardtanh(self.fc_speed(hds))
                speed_outputs    = torch.cat((speed_outputs, speed_output.unsqueeze(1)), dim = 1)
                in_sp            = speed_output.detach()
                
            outputs.append(speed_outputs)

        if 'intention' in self.args.task:
            crossing_outputs = torch.tensor([], device=self.args.device)
            in_cr = pos[:,-1,:]
            
            hdc = hpo + hsp
            cdc = cpo + csp
            for i in range(self.args.output//self.args.skip):
                hdc, cdc         = self.crossing_decoder(in_cr, (hdc, cdc))
                crossing_output  = self.fc_crossing(hdc)
                in_cr            = self.pos_embedding(hdc).detach()
                crossing_output  = self.softmax(crossing_output)
                crossing_outputs = torch.cat((crossing_outputs, crossing_output.unsqueeze(1)), dim = 1)

            outputs.append(crossing_outputs)
        
        if average:
            crossing_labels = torch.argmax(crossing_outputs, dim=2)
            intention = crossing_labels[:,-1]
            outputs.append(intention)
            
        return tuple(outputs)
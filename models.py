import torch
import torch.nn as nn
import torch.nn.functional as F


class Multitask_PS_LSTM(nn.Module):
    def __init__(self, args):
        super(Multitask_PS_LSTM, self).__init__()

        self.speed_encoder = nn.LSTM(input_size=4, hidden_size=args.hidden_size)
        self.pos_encoder   = nn.LSTM(input_size=4, hidden_size=args.hidden_size)

        self.pos_embedding = nn.Sequential(nn.Linear(in_features=2, out_features=4),
                                           nn.ReLU())

        self.speed_decoder    = nn.LSTMCell(input_size=4, hidden_size=args.hidden_size)
        self.intention_decoder = nn.LSTMCell(input_size=4, hidden_size=args.hidden_size)

        self.fc_speed    = nn.Linear(in_features=args.hidden_size, out_features=4)
        self.fc_crossing = nn.Sequential(nn.Linear(in_features=args.hidden_size, out_features=2),
                                         nn.ReLU())

        self.hardtan = nn.Hardtanh(min_val=-100, max_val=100)
        self.softmax = nn.Softmax(dim=1)

        self.args = args

    def forward(self, speed=None, pos=None):

        _, (speed_encoder_hidden, speed_encoder_cell) = self.speed_encoder(speed.permute(1,0,2))
        speed_encoder_hidden = speed_encoder_hidden.squeeze(0)
        speed_encoder_cell = speed_encoder_cell.squeeze(0)

        _, (pos_encoder_hidden, pos_encoder_cell) = self.pos_encoder(pos.permute(1,0,2))
        pos_encoder_hidden = pos_encoder_hidden.squeeze(0)
        pos_encoder_cell = pos_encoder_cell.squeeze(0)

        speed_outputs = torch.tensor([], device=self.args.device)
        last_speed = speed[:,-1,:]
        speed_decoder_hidden = pos_encoder_hidden + speed_encoder_hidden
        speed_decoder_cell = pos_encoder_cell + speed_encoder_cell

        intention_outputs = torch.tensor([], device=self.args.device)
        last_pos = pos[:,-1,:]
        intention_decoder_hidden = speed_decoder_hidden
        intention_decoder_cell = speed_decoder_cell

        for i in range(self.args.seq_len):
            speed_decoder_hidden, speed_decoder_cell = self.speed_decoder(last_speed, (speed_decoder_hidden, speed_decoder_cell))
            intention_decoder_hidden, intention_decoder_cell = self.intention_decoder(last_pos, (intention_decoder_hidden, intention_decoder_cell))

            speed_output = self.hardtan(self.fc_speed(speed_decoder_hidden))
            intention_output = self.fc_crossing(intention_decoder_hidden)

            last_speed = speed_output.detach()
            last_pos = self.pos_embedding(intention_output).detach()

            speed_outputs = torch.cat((speed_outputs, speed_output.unsqueeze(1)), dim = 1)
            crossing_outputs = torch.cat((intention_outputs, self.softmax(intention_output).unsqueeze(1)), dim = 1)

        return speed_outputs, crossing_outputs


class Speed_PS_LSTM(nn.Module):
    def __init__(self, args):
        super(Speed_PS_LSTM, self).__init__()

        self.pos_encoder = nn.LSTM(input_size=4, hidden_size=args.hidden_size)

        self.speed_encoder = nn.LSTM(input_size=4, hidden_size=args.hidden_size)

        self.decoder = nn.LSTMCell(input_size=4, hidden_size=args.hidden_size)

        self.fc = nn.Linear(in_features=args.hidden_size, out_features=4)

        self.activation = nn.Hardtanh(min_val=-100, max_val=100)

        self.args = args

    def forward(self, pos, speed):
        _, (pos_hidden, pos_cell) = self.pos_encoder(pos.permute(1,0,2))
        _, (speed_hidden, speed_cell) = self.speed_encoder(speed.permute(1,0,2))

        outputs = torch.tensor([], device=self.args.device)
        last_speed = speed[:,-1,:]

        pos_hidden = pos_hidden.squeeze(0)
        pos_cell = pos_cell.squeeze(0)
        speed_hidden = speed_hidden.squeeze(0)
        speed_cell = speed_cell.squeeze(0)

        decoder_hidden = pos_hidden+speed_hidden
        decoder_cell = pos_cell+speed_cell

        for i in range(self.args.seq_len):
            decoder_hidden, decoder_cell = self.decoder(last_speed, (decoder_hidden, decoder_cell))
            output = self.activation(self.fc(decoder_hidden))
            outputs = torch.cat((outputs, output.unsqueeze(1)), dim = 1)
            last_speed = output.detach()
        return outputs


class Intention_PS_LSTM(nn.Module):
    def __init__(self, args):
        super(Intention_PS_LSTM, self).__init__()

        self.speed_encoder = nn.LSTM(input_size=4, hidden_size=args.hidden_size)
        self.pos_encoder   = nn.LSTM(input_size=4, hidden_size=args.hidden_size)

        self.pos_embedding = nn.Sequential(nn.Linear(in_features=2, out_features=4),
                                           nn.ReLU())

        self.intention_decoder = nn.LSTMCell(input_size=4, hidden_size=args.hidden_size)

        self.fc_crossing = nn.Sequential(nn.Linear(in_features=args.hidden_size, out_features=2),
                                         nn.ReLU())
        self.softmax = nn.Softmax(dim=1)

        self.args = args

    def forward(self, speed=None, pos=None):

        _, (speed_hidden, speed_cell) = self.speed_encoder(speed.permute(1,0,2))
        speed_hidden = speed_hidden.squeeze(0)
        speed_cell = speed_cell.squeeze(0)

        _, (pos_hidden, pos_cell) = self.pos_encoder(pos.permute(1,0,2))

        pos_hidden = pos_hidden.squeeze(0)
        pos_cell = pos_cell.squeeze(0)

        crossing_outputs = torch.tensor([], device=self.args.device)
        last_pos = pos[:,-1,:]

        decoder_hidden = pos_hidden+speed_hidden
        decoder_cell = pos_cell+speed_cell
        for i in range(self.args.seq_len):
            decoder_hidden, decoder_cell = self.intention_decoder(last_pos, (decoder_hidden, decoder_cell))
            last_pos = self.pos_embedding(self.fc_crossing(decoder_hidden))
            crossing_output = self.softmax(self.fc_crossing(decoder_hidden))
            crossing_outputs = torch.cat((crossing_outputs, crossing_output.unsqueeze(1)), dim = 1)

        return crossing_outputs


class Speed_LSTM(nn.Module):
    def __init__(self, args):
        super(Speed_LSTM, self).__init__()

        self.encoder = nn.LSTM(input_size=4, hidden_size=args.hidden_size)

        self.decoder = nn.LSTMCell(input_size=4, hidden_size=args.hidden_size)

        self.fc = nn.Linear(in_features=args.hidden_size, out_features=4)

        self.activation = nn.Hardtanh(min_val=-100, max_val=100)

    def forward(self, input_):

        _, (h, c) = self.encoder(input_.permute(1,0,2))

        outputs = torch.tensor([], device=self.args.device)
        decoder_input = input_[:,-1,:]

        h = h.squeeze(0)
        c = c.squeeze(0)

        for i in range(self.args.seq_len):
            h, c = self.decoder(decoder_input, (h, c))
            output = self.activation(self.fc(h))
            outputs = torch.cat((outputs, output.unsqueeze(1)), dim = 1)
            decoder_input = output.detach()
        return outputs


class Pos_LSTM(nn.Module):
    def __init__(self, args):
        super(Speed_LSTM, self).__init__()

        self.encoder = nn.LSTM(input_size=4, hidden_size=args.hidden_size)

        self.decoder = nn.LSTMCell(input_size=4, hidden_size=args.hidden_size)

        self.fc = nn.Linear(in_features=args.hidden_size, out_features=4)

        self.activation = nn.ReLU()

    def forward(self, input_):

        _, (h, c) = self.encoder(input_.permute(1,0,2))

        outputs = torch.tensor([], device=self.args.device)
        decoder_input = input_[:,-1,:]

        h = h.squeeze(0)
        c = c.squeeze(0)

        for i in range(self.args.seq_len):
            h, c = self.decoder(decoder_input, (h, c))
            output = self.activation(self.fc(h))
            outputs = torch.cat((outputs, output.unsqueeze(1)), dim = 1)
            decoder_input = output.detach()
        return outputs


class Scene_PS_LSTM(nn.Module):
    def __init__(self, args):
        super(Scene_PS_LSTM, self).__init__()
        self.basenet = torchvision.models.resnet50(pretrained=True)
        self.basenet = nn.Sequential(*list(self.basenet.children())[:-2])
        self.basenet = nn.Sequential(self.basenet, nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1))

        self.speed_encoder = nn.LSTM(input_size=4, hidden_size=args.hidden_size)

        self.pos_encoder = nn.LSTM(input_size=4, hidden_size=args.hidden_size)

        self.scene_encoder = nn.LSTM(input_size=512*8*14, hidden_size=args.hidden_size)

        self.decoder = nn.LSTMCell(input_size=4, hidden_size=args.hidden_size)

        self.fc = nn.Linear(in_features=args.hidden_size, out_features=4)

        self.activation = nn.Hardtanh(min_val=-100, max_val=100)

        self.args = args

    def forward(self, scenes, speed, pos):

        scenes  = self.basenet(scenes.view(scenes.shape[0]*scenes.shape[1], scenes.shape[2],
                                           scenes.shape[3], scenes.shape[4]))

        _, (scene_hidden, scene_hidden) = self.scene_encoder(scenes.view(self.args.batch_size, self.args.seq_len, -1).permute(1,0,2))

        _, (speed_hidden, speed_cell) = self.speed_encoder(speed.permute(1,0,2))

        _, (pos_hidden, pos_cell) = self.pos_encoder(pos.permute(1,0,2))

        outputs = torch.tensor([], device=self.args.device)
        last_speed = speed[:,-1,:]

        decoder_hidden = scene_hidden + speed_hidden + pos_hidden
        decoder_cell = scene_hidden + speed_cell + pos_cell

        decoder_hidden = decoder_hidden.squeeze(0)
        decoder_cell = decoder_cell.squeeze(0)

        for i in range(self.args.seq_len):
            decoder_hidden, decoder_cell = self.decoder(last_speed, (decoder_hidden, decoder_cell))
            output = self.activation(self.fc(decoder_hidden))
            outputs = torch.cat((outputs, output.unsqueeze(1)), dim = 1)
            last_speed = output.detach()

        return outputs

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import logistic
import argparse

DEF_ARGS = argparse.Namespace(data_path='ex_seqs.fa', output_path='ex_seqs_out', cuda=False)

class ORNet(nn.Module):
    def __init__(self, in_features, nclass, device):
        super(ORNet, self).__init__()
        # first store constants
        self.nclass = nclass # number of classes
        self.in_features = in_features
        self.device = device
        # initialize bias parameters for or ordinal regression model
        biasm = torch.randn(self.nclass-1)
        #self.or_bias = biasm.sort(descending=True)[0]
        self.or_bias = torch.nn.Parameter(biasm.sort(descending=True)[0])
        self.fc1_bias = nn.Linear(self.in_features, 1, bias=False)
        self.logit = False
        
    def forward(self, x):
        # calculate t(W_or,i)*Phi(x)
        x = self.fc1_bias(x)
        # now split among last dimension
        out = []
        # run over each class
        for i,wp_i in enumerate(range(self.nclass-1)):
            if self.logit:
                temp = self.or_bias[i]+x
            else:
                temp = torch.sigmoid(self.or_bias[i]+x)
            #temp = 1/(1+torch.exp(-(x+self.or_bias[i])))
            # now we run over each couple and subtract consecutive values
            out.append(temp)
        y = torch.cat(out,1)
        x = torch.cat((torch.ones((y.shape[0],1)).to(self.device), y[:],
                       torch.zeros((y.shape[0],1)).to(self.device)), dim=1)
        x = x[:,:-1] - x[:,1:]
        x = torch.log(torch.div(x,(1-x)))
        
        if self.logit:
            #y = torch.log(torch.div(y,(1-y)))
            return y
        else:
            return x

class PromNeural(nn.Module):
    def __init__(self, nclass, batch_size, device):
        """The DeepRibo model architecture

        Arguments:
            hidden_size (int): weights allocated to the GRU
            layers (int): amount of GRU layers
            bidirect (bool): model uses a bidirectional GRU
        """
        super(PromNeural, self).__init__()
        self.dev = device
        self.or_type = 3
        self.layers = 2
        self.do = nn.Dropout(0.3)
        self.nclass = nclass # number of classes   
        
        self.conv_kernels = 16
        self.conv_kernels_2 = 32
        
        self.conv_ch_1 = nn.Conv2d(4, 4, (1, 1))
        self.conv1 = nn.Conv2d(4, self.conv_kernels, (4, 1), stride=(1,1))
        self.conv2 = nn.Conv2d(self.conv_kernels, self.conv_kernels_2,
                               (2, 1), stride=(1,1))
        self.fc1 = nn.Linear(13*self.conv_kernels_2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.ornet = ORNet(64, self.nclass, self.dev)

    def forward(self, x, hidden=None):
        x = F.relu(self.conv_ch_1(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 13*self.conv_kernels_2)
        x = F.relu(self.fc1(self.do(x)))
        x = F.relu(self.fc2(x))
        
        x = self.ornet(x)
        
        return x

def initialize_model(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    nclass = 11
    batch_size = 32
    model = PromNeural(nclass, batch_size, device)
    model.load_state_dict(torch.load('models/model_RPOD.pt', map_location=device))
    model.to(device)
    model.eval()
    
    return model

def read_data(args):
    nts = ['A', 'C', 'G', 'T']
    seqs = []
    seq_ids = []
    with open(args.data_path) as f:
        for line in f:
            seq = line.strip().upper()
            if seq[0] == '>':
                seq_id = seq[1:]
            elif (np.array([nt in nts for nt in list(seq)]).all()) and (len(seq) == 17):
                seqs.append(seq)
                seq_ids.append(seq_id)
            else:
                print(f'{seq} is not a valid spacer sequence and is excluded.')
    assert len(seqs)>0, "No valid sequences"
    
    return seq_ids, seqs

def transform_data(seqs):
    seq_dict = {"A": 0, "T": 1, "C": 2, "G": 3}
    seq_img = np.full((len(seqs), 4, len(seqs[0]), 1), 0)
    for i, string in enumerate(seqs):
        for j, nt in enumerate(string):
            if nt in ['A','T','C','G']:
                seq_img[i, seq_dict[nt], j, 0] = 1
                
    return seq_img

def create_output(pred_te_prob, pred_te_disc, seq_ids, seqs, args):
    probs_dict = {f'P(Class {i}|spacer)':probs for i,probs in enumerate(pred_te_prob.T)}
    if len(seq_ids) == len(seqs):
        ids = seq_ids
    else:
        ids = np.arange(len(seqs))

    output = pd.DataFrame({'ID': ids, 'spacer':seqs, 'strength': pred_te_disc})

    for key, series in probs_dict.items():
        output[key] = series
    output.to_csv(f'{args.output_path}.csv')
    
    return output

def forward_pass(model, seq_img, args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    pred_te = model(torch.Tensor(seq_img).to(device))
    pred_te_prob = logistic.cdf(pred_te.cpu().data.numpy())
    pred_te_disc = np.argmax(pred_te_prob, axis=1).astype(np.int)
    
    return pred_te_prob, pred_te_disc

def main(args):

    
    # INITIALIZE MODEL
    model = initialize_model(args)

    # READ DATA
    seq_ids, seqs = read_data(args)

    # TRANSFORM DATA
    seq_img = transform_data(seqs)

    # FORWARD PASS
    pred_te_prob, pred_te_disc = forward_pass(model, seq_img, args)
    
    # CREATE OUTPUT
    output = create_output(pred_te_prob, pred_te_disc, seq_ids, seqs, args)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Promoter Strength Prediction tool')
    parser.add_argument('data_path', type=str, help='location of the text file containing spacer sequences')
    parser.add_argument('--output_path', '-o', type=str, default='output',
                        help='location of the text file containing spacer sequences')
    parser.add_argument('--cuda', action='store_true', help='use CUDA. Requires PyTorch installation supporting CUDA!')
    args = parser.parse_known_args()[0]
    if args.output_path[-4:] == '.csv':
        args.output_path = args.output_path[:-4]
        
    main(args)
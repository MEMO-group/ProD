import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import logistic
import argparse
from pdb import set_trace

DEF_ARGS = argparse.Namespace(data_path='ex_seqs.fa', output_path='ex_seqs_out', cuda=False)

SEQ_DICT = {'A': [0], 'T': [1], 'C': [2], 'G': [3], 'R': [0,3],
            'Y': [1,2], 'S': [2,3], 'W': [0,1], 'K': [1,3],
            'M': [0,2], 'B': [1,2,3], 'D': [0,1,3], 
            'H': [0,1,2], 'V': [0,2,3], 'N':[0,1,2,3]}
SS_DICT= {'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'], 'R': ['A','G'],
            'Y': ['T','C'], 'S': ['C','G'], 'W': ['A','T'], 'K': ['T','G'],
            'M': ['A','C'], 'B': ['T','C','G'], 'D': ['A','T','G'], 
            'H': ['A','T','C'], 'V': ['A','C','G'], 'N':['A','T','C','G']}
IMG_DICT = {0: 'A', 1:'T', 2:'C', 3:'G'}

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

def initialize_model(cuda=False):
    device = torch.device('cuda' if cuda else 'cpu')
    nclass = 11
    batch_size = 32
    model = PromNeural(nclass, batch_size, device)
    model.load_state_dict(torch.load('models/model_RPOD.pt', map_location=device))
    model.to(device)
    model.eval()
    
    return model

def parse_lines(lines):
    seqs = []
    seq_ids = []
    excl_seqs = []
    for line in lines:
        seq = line.strip().upper()
        if seq[0] == '>':
            seq_id = seq[1:]
            seq_ids.append(seq_id)
        elif (np.array([nt in SEQ_DICT.keys() for nt in list(seq)]).all()) and (len(seq) == 17):
            seqs.append(seq)
        else:
            excl_seqs.append(seq)
        
    if len(excl_seqs) > 0:
        print(f'{excl_seqs}: invalid spacer sequences excluded from data.')
    
    return seq_ids, seqs

def read_data(input_data):
    assert type(input_data) in [str, list, np.ndarray], f'input data: {type(input_data)} not accepted'
    if type(input_data) is str:
        with open(input_data) as f:
            seq_ids, seqs = parse_lines(f)
    else:
        seq_ids, seqs = parse_lines(input_data)
    assert len(seqs)>0, "No valid sequences"
    
    return seq_ids, seqs

def string_to_img(seqs):
    seqs_img = np.full((len(seqs), 4, len(seqs[0]), 1), 0)
    for idx1, string in enumerate(seqs):
        for idx2, nt in enumerate(string):
            assert nt in SEQ_DICT.keys(), f'invalid input: {nt}'
            seqs_img[idx1, np.random.choice(SEQ_DICT[nt]), idx2, 0] = 1
                
    return seqs_img

def string_to_string(seqs):
    seqs_new = []
    seq = np.full(len(seqs[0]), 'N',dtype='|S1')
    for string in seqs:
        for idx, nt in enumerate(string):
            assert nt in SS_DICT.keys(), f'invalid input: {nt}'
            seq[idx] = np.random.choice(SS_DICT[nt])
        seqs_new.append(seq.tostring().decode('utf-8'))
                
    return seqs_new

def img_to_string(seqs_img):
    seqs = []
    seq = np.full(seqs_img[0].shape[1], 'N',dtype='|S1')
    for seq_img in seqs_img:
        for idx, nt_img in enumerate(seq_img.transpose(1,0,2)):
            seq[idx] = IMG_DICT[nt_img.argmax()]
        seqs.append(seq.tostring().decode('utf-8'))
    
    return seqs

def forward_pass(model, seq_img, cuda=False):
    device = torch.device('cuda' if cuda else 'cpu')
    pred_te = model(torch.Tensor(seq_img).to(device))
    pred_te_prob = logistic.cdf(pred_te.cpu().data.numpy())
    pred_te_disc = np.argmax(pred_te_prob, axis=1).astype(np.int)
    
    return pred_te_prob, pred_te_disc

def create_output(pred_te_prob, pred_te_disc, seq_ids, seqs):
    probs_dict = {f'P(Class {i}|spacer)':probs for i,probs in enumerate(pred_te_prob.T)}
    if len(seq_ids) == len(seqs):
        ids = seq_ids
    else:
        ids = np.arange(len(seqs))

    output = pd.DataFrame({'ID': ids, 'spacer':seqs, 'strength': pred_te_disc})
    
    output['promoter'] = ' GGTCTATGAGTGGTTGCTGGATAAC TTTACG ' + output.spacer + \
    ' TATAAT ATATTC AGGGAGAGCACAACGGTTTCCCTCTACAAATAATTTTGTTTAACTTT'

    for key, series in probs_dict.items():
        output[key] = series
    
    return output

def forward(input_data, lib=False, lib_size=5, cuda=False):
    # INITIALIZE MODEL
    model = initialize_model(cuda)
    # READ DATA
    seq_ids, seqs = read_data(input_data)
    if lib:
        print(f'Using {seqs[0]} as blueprint')
        total = 1
        for idx in np.arange(len(seqs[0])):
            total *= len(SEQ_DICT[seqs[0][idx]])
        print(f'{total} sequences possible, sampling {np.minimum(total, 5000)}')
        # GENERATE DATA
        seqs = np.full(10000, seqs[0])
    # TRANSFORM DATA    
    seqs = np.unique(string_to_string(seqs))
    seqs_img = string_to_img(seqs)
    
    idx = 0
    outputs = []
    while idx*1000 < len(seqs):
        # FORWARD PASS
        pred_prob, pred_disc = forward_pass(model, seqs_img[idx*1000:(idx+1)*1000], cuda)
        # CREATE OUTPUT
        outputs.append(create_output(pred_prob, pred_disc,
                               seq_ids[idx*1000:(idx+1)*1000], 
                               seqs[idx*1000:(idx+1)*1000]))
        idx += 1
        if lib:
            out_temp = pd.DataFrame().append(outputs, ignore_index=True)
            if (out_temp.strength.value_counts() >= lib_size).all():
                break
            elif idx*1000 >= len(seqs):
                print(f'Could not find requested size for each of the classes.')
                if total>5000:
                    print(f'Sampled 5000 out of {total} possible sequences.\
                    Rerun the tool to evaluate more samples')
    
    outputs = pd.DataFrame().append(outputs, ignore_index=True)
    if lib:
        output_list = [outputs.loc[outputs.strength == i][:lib_size] for i in np.arange(11)]
        outputs = pd.DataFrame().append(output_list, ignore_index=True)  
    return outputs
    
def run_tool(input_data, output_path='my_predictions', lib=False, lib_size=5, cuda=False):
    output = forward(input_data, lib, lib_size, cuda)
    output.to_csv(f'{output_path}.csv')
    
    return output.iloc[:,:4]
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Promoter Designer (ProD) tool')
    parser.add_argument('data_path', type=str, help='location of the text file containing spacer sequences')
    parser.add_argument('--output_path', '-o', type=str, default='output',
                    help='location of the text file containing spacer sequences')
    parser.add_argument('--lib', action='store_true', help='create library from blueprint')
    parser.add_argument('--lib_size', '-ls', type=int, default='5',
                    help='size of each class in library')
    parser.add_argument('--cuda', action='store_true', help='use CUDA. Requires PyTorch installation supporting CUDA!')
    args = parser.parse_known_args()[0]
    if args.output_path[-4:] == '.csv':
        args.output_path = args.output_path[:-4]
    run_tool(args.data_path, args.output_path, args.lib, args.lib_size, args.cuda)
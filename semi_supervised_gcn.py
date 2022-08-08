import numpy as np
import scipy.sparse as sp
import time
import torch
import math
import torch.optim as optim
import time
import argparse
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F



data_path='C:/Users/ydb80/Desktop/GNN/pygcn-master/data/cora/'

def load_data(path=data_path, dataset="cora"):
    """citation network cora dataset load"""
    print('Loading {} dataset...'.format(dataset))

    # index, features, labels load
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),  #텍스트 파일을 np array형태로  -content로 끝나는 파일
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1],
                             dtype=np.float32)  # 각 node의 feature를 추출함, sparse한 matrix 효율적 처리
    labels = encode_onehot(idx_features_labels[:, -1])  # label 정보 one hot encoding

    # build graph
    # 각 노드의 index 부여 2708개
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32) #논문 번호
    idx_map = {j: i for i, j in enumerate(idx)}  #논문 번호, 행번호 순서

    # 각 노드 간 연결 edge 데이터를 load하고 index로 다시 부여함
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),  #np.flatten() 일차원으로 변경해줌.[1,2][3,4]--> [1,2,3,4]
                     dtype=np.int32).reshape(edges_unordered.shape)  #각 논문을 행번호로 노드간 표현
    # adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    # input data: A + In
    adj = normalize(adj + sp.eye(adj.shape[0]))  #sp.eye -대각이 1, 나머지는 0인 행렬 

    # mask index for semi-supervised node classification
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))  #todesnse 희소행렬을 다시 array형태로 변환할수 있게 해줌, 3차원을 tensor라함. floattensor는 tensor형태로 만들어 준다는것. 1,2차원도 텐서라고 하기도함
    labels = torch.LongTensor(np.where(labels)[1])  #longtensor--> 64bit의 integer로 변환 
    adj = sparse_mx_to_torch_sparse_tensor(adj)  # torch tensor type 변환

    idx_train = torch.LongTensor(idx_train) #0부터 139까지 tensor형태로 변환
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


adj, features, labels, idx_train, idx_val, idx_test=load_data()






def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in  # np.identity 정방 행렬 만듦
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-wise normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  #행끼리 더함 
    r_inv = np.power(rowsum, -1).flatten() #  #행sum에 inverse
    r_inv[np.isinf(r_inv)] = 0.#무한대로 가면 0으로 수정
    r_mat_inv = sp.diags(r_inv) #대각행렬로 만들어줌
    mx = r_mat_inv.dot(mx)  #대각 행렬의 곱
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """scipy sparse matrix -> torch sparse tensor"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphConvolution(Module):
    """GCN layer"""

    # normalization은 기존에 했기 때문에 생략함
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameter W 정의
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))  #.size() 전체 원소의 개수 반환
        self.weight.data.uniform_(-stdv, stdv)  #weight를 -stdv,stdv 사이로 균일하게 
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # input feature와 W 계산
        support = torch.mm(input, self.weight)  # input과 weigt의 행렬곱 수행

        # A와 XW 계산
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):  #신경망 뼈대상속
    def __init__(self, num_features, num_hidden, num_class, dropout):
        super(GCN, self).__init__()  

        # 각 정의된 layer를 불러와 GCN 연산함
        # 입력 features 수, 1433
        self.gc1 = GraphConvolution(num_features, num_hidden)
        self.gc2 = GraphConvolution(num_hidden, num_class)
        self.dropout = dropout

    def forward(self, x, adj): #처음에 x=features
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        # log_softmax 취함
        x = F.log_softmax(x, dim=1)

        return x


def train(epoch):
    t = time.time()
    model.train() #train time으로 switching
    optimizer.zero_grad()
    output = model(features, adj)

    # 앞서 log_softmax를 취했기 때문에 nll_loss 적용함
    # semi-supervised 접근을 위해 train의 index만 loss 계산하고 측정함
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step() #loss 효율적이게 할수 있도록 넘김

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval() #evaluation time-evaluation 과정에서 사용하지 말아야할 layer들 알아서 off시킴
        output = model(features, adj)

    # validation loss 계산
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    # test loss 및 accuracy 계산
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


class Args:
    fastmode = False
    seed = 42
    epochs = 200
    lr = 0.01
    weight_decay = 5e-4
    hidden = 16
    dropout = 0.5
#     cuda = 'cuda:0'



if args.cuda:  #cuda사용하면..
    torch.cuda.manual_seed(args.seed)


# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(num_features=features.shape[1],
            num_hidden=args.hidden,
            num_class=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),  #adam algorithm
                       lr=args.lr, weight_decay=args.weight_decay)

#if args.cuda:
 #   model.cuda()
  #  features = features.cuda()
  #  adj = adj.cuda()
  #  labels = labels.cuda()
   # idx_train = idx_train.cuda()
   # idx_val = idx_val.cuda()
   # idx_test = idx_test.cuda()

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

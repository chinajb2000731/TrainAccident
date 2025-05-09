import random
import numpy as np
import torch
import tqdm
from torch import  nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch import Tensor
import jieba

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
def sequence_mask(x,valid_len,value=0):
    '''屏蔽pad项'''
    maxlen=x.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=x.device)[None, :] < valid_len[:, None]
    x = torch.where(~mask, value, x)
    # x[~mask]=value
    return x


def mask_softmax(x,valid_lens):
    '''屏蔽pad，将其softmax后的值置为0'''
    if valid_lens is None:
        return nn.functional.softmax(x,dim=-1)
    else:
        shape=x.shape
        if valid_lens.dim()==1:
            valid_lens=torch.repeat_interleave(valid_lens,shape[1])
        else:
            valid_lens=valid_lens.reshape(-1)
    x=sequence_mask(x.reshape(-1,shape[-1]),valid_lens,value=-1e6)
    return nn.functional.softmax(x.reshape(shape),dim=-1)


def transpose_qkv(x,num_heads):
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    x=x.reshape(x.shape[0],x.shape[1],num_heads,-1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    x=x.permute(0,2,1,3)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,num_hiddens/num_heads)
    return x.reshape(-1,x.shape[2],x.shape[3])


def transpose_output(x,num_heads):
    x=x.reshape(-1,num_heads,x.shape[1],x.shape[2])
    x=x.permute(0,2,1,3)
    return x.reshape(x.shape[0],x.shape[1],-1)
def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1:
        return torch.device(f"cuda:{i}")
    return torch.device('cpu')


def grad_clipping(net, theta):
    """梯度裁剪函数"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def correct_num(y_hat,y):
    num=0
    correct=0
    for i in range(len(y)):
        if y[i]==y_hat[i]:
            correct+=1
        num+=1
    return num,correct

def correct_num_predict(y_hat,y):
    if len(y)!=len(y_hat):
        return 0
    else:
        for i in range(len(y)):
            if y[i] not in y_hat[i]:
                return 0
    return 1


def Train_Model(net,train_data_iter,valid_data_iter,lr,num_epochs,device):
    def xavier_init_weights(m):
        if type(m)==nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m)==nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)
        if type(m)==nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)

    optimizer = torch.optim.Adam([
        {'params': net.encoder.embedding.parameters(), 'lr': 0.0002},
        {'params': net.gru1.parameters(), 'lr': 0.0001},
        {'params': net.gru2.parameters(), 'lr': 0.0001},
        {'params': net.encoder.blks.parameters()},
    ], lr=lr)
    # optimizer=torch.optim.Adam([
    #     {'params': net.encoder.embedding.parameters(),'lr':0.0002},
    #     {'params': net.gru1.parameters(),'lr':0.0001},
    #     {'params': net.gru2.parameters(),'lr':0.0005},
    #     {'params': net.encoder.blks.parameters()},
    # ],lr=lr)
    loss=nn.CrossEntropyLoss()
    max_valid_accuracy = 0
    for epoch in range(num_epochs):
        net.train()
        sum_loss=0
        sum_tokens=0
        nums=0
        corrects=0
        for batch in tqdm.tqdm(train_data_iter):
            optimizer.zero_grad()
            x,x_valid_len,y=[i.to(device) for i in batch]

            y_hat=net(x,x_valid_len)


            l=loss(y_hat,y)
            l.sum().backward()
            num_tokens=len(y)
            # grad_clipping(net,1)
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=2)
            optimizer.step()

            with torch.no_grad():
                sum_tokens+=num_tokens
                sum_loss+=l.sum()
                y_hat=y_hat.argmax(dim=1)
                num,correct=correct_num(y_hat,y)
                nums+=num
                corrects+=correct
        print(f'epoch{epoch + 1}:loss{sum_loss / sum_tokens:.5f},训练集正确率{corrects / nums},corrects={corrects},nums={nums}')
        with open("./loss_trans_cnn_gru.txt", "a", encoding="utf-8") as f:
            f.write(str(sum_loss / sum_tokens))
            f.write("\n")
            f.close()
        # 训练集正确率
        with open("./train_accuracy_trans_cnn_gru.txt", "a", encoding="utf-8") as f:
            f.write(str(corrects / nums))
            f.write("\n")
            f.close()
        # 进行验证集测试
        # if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            precision, recall, f1, accurcay = valid_Macro(net, valid_data_iter, device)
            print(f'epoch{epoch + 1}:验证集：accurcay={accurcay:.5f},precision={precision:.5f},recall={recall:.5f},f1={f1:.5f}')
            if max_valid_accuracy < accurcay:
                max_valid_accuracy = accurcay
                torch.save(net.state_dict(), './checkpoint/net_trans_gru_{}.pth'.format(max_valid_accuracy))
            with open("./valid_accuracy_trans_cnn_gru.txt", "a", encoding="utf-8") as f:
                f.write(str(accurcay) + " " + str(precision) + " " + str(recall) + " " + str(f1) + " ")
                f.write("\n")
                f.close()
        if ((epoch+1))<=10 and ((epoch + 1)) % 5 == 0:
            print("学习率降低........")
            for p in optimizer.param_groups:
                p['lr'] *= 0.7  # 注意这里
        elif (epoch+1)>10:
            for p in optimizer.param_groups:
                p['lr']*=0.9

def valid(net,valid_set,device):
    net.eval()
    nums=0
    correct=0
    for batch in tqdm.tqdm(valid_set):
        x, x_valid_len, y = [i.to(device) for i in batch]
        y_hat=net(x,x_valid_len)
        correct+=(y==y_hat.argmax(1)).sum()
        nums+=len(y)
    return correct,nums

def valid_Macro(net,valid_set,device):
    net.eval()
    y_hat_list=[]
    y_list=[]
    for batch in tqdm.tqdm(valid_set):
        x, x_valid_len, y = [i.to(device) for i in batch]
        y_hat=net(x,x_valid_len)
        y_hat_list.extend(item.cpu().numpy() for item in y_hat.argmax(1))
        y_list.extend(item.cpu().numpy() for item in y)
    precision = precision_score(y_list,y_hat_list, average='weighted')
    recall = recall_score(y_list, y_hat_list, average='weighted')
    f1 = f1_score(y_list, y_hat_list, average='weighted')
    accurcay = accuracy_score(y_list, y_hat_list)
    return precision,recall,f1,accurcay

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self,pred,label,valid_len):
        weights=torch.ones_like(label)
        weights=sequence_mask(weights,valid_len)
        self.reduction='none'
        unweighted_loss=super(MaskedSoftmaxCELoss,self).forward(pred.permute(0,2,1),label)
        weighted_loss=(unweighted_loss*weights).mean(dim=1)
        return weighted_loss

def truncate_pad(line,num_steps,padding_token):
    '''文本填充与截断'''
    if len(line)>num_steps:
        return line[:num_steps]
    return line+[padding_token]*(num_steps-len(line))

#-*- coding:utf-8 -*-
import collections
import json
from  torch.utils.data import Dataset
from Word2Vec.word2vec import *
import tqdm


def tokenize(read_path):
    file_lines = open(read_path, "r", encoding="utf-8").readlines()
    facts = list() #数据集的事实描述切词
    causes = list()  # 原因
    for line in tqdm.tqdm(file_lines):
        events = json.loads(line)
        causes.append(events['label'])
        fact = events['splice_text'].split(" ")
        facts.append(fact)

    return facts,causes


def truncate_pad(line,num_steps,padding_token,valid_lens):
    '''文本填充与截断'''
    if len(line)>num_steps:
        valid_lens.append(num_steps)
        return line[:num_steps],valid_lens
    else:
        valid_lens.append(len(line))
    return line+[padding_token]*(num_steps-len(line)),valid_lens

def build_data(lines,num_steps):
    valid_lens = []
    lines=[l+['<EOS>'] for l in tqdm.tqdm(lines)]
    lists=[]
    for line in tqdm.tqdm(lines):
        sen,valid_lens=truncate_pad(line,num_steps,'<PAD>',valid_lens)
        lists.append(sen)
    return lists,valid_lens

#62分类
class RailDataset(Dataset):
    #这里的data是切好词的数据,data
    def __init__(self,data_path,num_steps,vocab):
        super(RailDataset, self).__init__()
        self.num_steps=num_steps
        fact_tokens,cause_tokens=tokenize(data_path)
        self.vocab_facts = vocab  # 事实描述词表，这个词表是词向量训练过后的

        self.facts,self.facts_valid_lens=build_data(fact_tokens,num_steps)#pad过后的数据
        self.causes=cause_tokens

    def __getitem__(self, item):
        return torch.tensor(self.vocab_facts[self.facts[item]]),torch.tensor(self.facts_valid_lens[item]), torch.tensor(self.causes[item])
    def __len__(self):
        return len(self.facts) #数据量大小



if __name__ == '__main__':
    # facts,causes=tokenize("../Data/Data_train/valid_label.json")
    # print(type(causes[0]))
    # vocab=my_Vocab(causes)
    # print(vocab.token_to_idx)
    # print(facts)
    # print(causes)
    # print(facts[14])
    vocab=EmbGensim("../Word2Vec/word2vec_128.model")
    train_data_set=RailDataset("../Data/Data_train/valid_label.json",200,vocab)
    # print(train_data_set.vocab_causes.token_to_idx)
    a,b,c=train_data_set[0]
    print(a)
    print(b)
    print(c)
    print()

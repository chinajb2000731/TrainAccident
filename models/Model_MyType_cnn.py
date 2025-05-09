from DataSet import *
from EncoderDecoder import *
from utils_cnn import *
from  torch.utils.data import DataLoader
path="../Data/Data_MyType/train.json"
valid_path="../Data/Data_MyType/valid.json"
emb_pat="../Word2Vec/word2vec_128.model"

seed_everything()
vocab=EmbGensim(emb_pat)

train_data_set=RailDataset(path,256,vocab)
valid_data_set=RailDataset(valid_path,256,vocab)
train_data_iter=DataLoader(train_data_set,200,shuffle=True)
valid_data_iter=DataLoader(valid_data_set,200)

embeds=vocab.getVectors()

encoder=TransformerEncoderCnn(len(vocab),128,128,128,128,128,8,[128],128,256,128,4,0.2)
encoder.embedding.weight.data.copy_(embeds)


net=EncoderCnn2(encoder,128,[4,8,16,32,64,128,256],[62,62,62,62,62,62,62])
print(net)
print(sum([param.nelement() for param in net.parameters()]))
Train_Model(net,train_data_iter,valid_data_iter,0.0002,30,try_gpu())


"""Two complementary small neural architectures, both trained from scratch on names.

FP32 parameters are exported losslessly as JSON numbers (no weights quantized).
"""
from __future__ import annotations
import copy
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

MAX_LEN = 10


def vocab_from_names(names, limit=3600):
    freq=Counter(ch for name in names for ch in name[:MAX_LEN])
    out={'<PAD>':0,'<UNK>':1}
    for ch,_ in sorted(freq.items(),key=lambda pair:(-pair[1],pair[0]))[:limit-2]:
        out[ch]=len(out)
    return out


def encode(names, vocab, max_len=MAX_LEN):
    result=np.zeros((len(names),max_len),dtype=np.int64)
    sizes=np.empty((len(names),),dtype=np.int64)
    for i,name in enumerate(names):
        chars=list(name)[:max_len]; sizes[i]=max(1,len(chars))
        for j,ch in enumerate(chars):result[i,j]=vocab.get(ch,1)
    return result,sizes


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, emb=128, channels=96):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb,padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(emb,channels,k) for k in (2,3,4)])
        self.dropout=nn.Dropout(0.15)
        self.head=nn.Linear(channels*3,num_classes)

    def forward(self, tokens, lengths):
        x=self.embedding(tokens).transpose(1,2)
        pooled=[]
        for k,conv in zip((2,3,4),self.convs):
            z=torch.relu(conv(x))
            valid=torch.clamp(lengths-k+1,min=1)
            pos=torch.arange(z.size(-1),device=z.device)
            z=z.masked_fill(pos[None,None,:]>=valid[:,None,None],-1e5)
            pooled.append(z.max(dim=-1).values)
        return self.head(self.dropout(torch.cat(pooled,dim=-1)))


class BiGRU(nn.Module):
    def __init__(self,vocab_size,num_classes,emb=128,hidden=128):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb,padding_idx=0)
        self.rnn=nn.GRU(emb,hidden,batch_first=True,bidirectional=True)
        self.dropout=nn.Dropout(0.15)
        self.head=nn.Linear(hidden*2,num_classes)

    def forward(self,tokens,lengths):
        emb=self.embedding(tokens)
        packed=pack_padded_sequence(emb,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,state=self.rnn(packed)
        return self.head(self.dropout(torch.cat([state[0],state[1]],dim=-1)))


MODELS={'cnn':TextCNN,'bigru':BiGRU}


def probabilities(model,names,vocab,batch=1024):
    model.eval(); result=[]
    with torch.inference_mode():
        for start in range(0,len(names),batch):
            x,l=encode(names[start:start+batch],vocab)
            output=model(torch.from_numpy(x),torch.from_numpy(l))
            result.append(torch.softmax(output,dim=-1).numpy())
    if not result:return np.empty((0,model.head.out_features),dtype=np.float32)
    return np.concatenate(result).astype(np.float64)


def train_neural(architecture, rows, classes, val_rows, seed, *,
                 per_class=20000, epochs=7, batch_size=256, vocab_limit=3600,
                 lr=0.0015, emb_dim=128, cnn_channels=96, gru_hidden=128, smoke=False):
    assert architecture in MODELS
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    rng=np.random.default_rng(seed)
    selected=[]
    for lab in classes:
        group=[r for r in rows if r['label']==lab]
        priority=[r for r in group if r.get('tier')=='hard_case']
        others=[r for r in group if r.get('tier')!='hard_case']
        if per_class:
            if len(priority)>per_class:
                raise ValueError('hard-case quota exceeds neural per-class budget')
            quota=max(0,per_class-len(priority))
            if len(others)>quota:
                indices=rng.choice(len(others),size=quota,replace=False)
                others=[others[int(i)] for i in indices]
        selected.extend(priority)
        selected.extend(others)
    rng.shuffle(selected)
    names=[r['name'] for r in selected]
    y=np.array([classes.index(r['label']) for r in selected],dtype=np.int64)
    if len(set(y))!=len(classes):raise ValueError('A neural training class is absent')
    vocab=vocab_from_names(names,vocab_limit)
    x,l=encode(names,vocab)
    dataset=TensorDataset(torch.from_numpy(x),torch.from_numpy(l),torch.from_numpy(y))
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    model=(TextCNN(len(vocab),len(classes),emb=emb_dim,channels=cnn_channels)
           if architecture=='cnn' else BiGRU(len(vocab),len(classes),emb=emb_dim,hidden=gru_hidden))
    counts=np.bincount(y,minlength=len(classes)).astype(np.float32)
    weights=np.sqrt(np.max(counts)/np.maximum(counts,1.0))
    weights=np.minimum(weights,4.0)
    weights=weights/weights.mean()
    objective=nn.CrossEntropyLoss(weight=torch.tensor(weights),label_smoothing=0.02)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    val_sample=val_rows
    if len(val_sample)>6000:
        ids=np.sort(rng.choice(len(val_sample),size=6000,replace=False))
        val_sample=[val_sample[int(i)] for i in ids]
    val_names=[r['name'] for r in val_sample]
    val_y=[classes.index(r['label']) for r in val_sample]
    best=-1.0; state=None; history=[]; bad=0
    for epoch in range(1,epochs+1):
        model.train(); total=0.; count=0
        for inputs,lengths,targets in loader:
            opt.zero_grad(set_to_none=True)
            loss=objective(model(inputs,lengths),targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            opt.step()
            total+=float(loss.item())*len(targets); count+=len(targets)
        p=probabilities(model,val_names,vocab)
        score=float(f1_score(val_y,p.argmax(1),labels=range(len(classes)),average='macro',zero_division=0))
        history.append({'epoch':epoch,'loss':round(total/max(count,1),6),'val_macro_f1':round(score,6)})
        print(f'[{architecture}] epoch={epoch}/{epochs} loss={total/max(count,1):.4f} val_macro_f1={score:.4f}',flush=True)
        if score>best+1e-4:
            best=score; state=copy.deepcopy(model.state_dict()); bad=0
        else:
            bad+=1
            if bad>=3 and not smoke:break
    if state is None: raise RuntimeError('No best neural checkpoint')
    model.load_state_dict(state)
    exported={'arch':architecture,'vocab':vocab,'max_len':MAX_LEN,'classes':classes,
              'hyperparameters':{'emb':emb_dim,'channels':cnn_channels,'hidden':gru_hidden},
              'state':{k:{'shape':list(v.shape),'values':v.detach().cpu().numpy().astype(np.float32).flatten().tolist()}
                       for k,v in model.state_dict().items()}}
    return model,vocab,exported,{'used_training_samples':len(names),'history':history,
                                 'best_epoch':max(history,key=lambda h:h['val_macro_f1'])['epoch'],
                                 'best_val_macro_f1':best,'parameter_count':sum(p.numel() for p in model.parameters())}

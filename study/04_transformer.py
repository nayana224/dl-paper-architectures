"""Attention Is All You Need(2017): 인코더·디코더, 마스킹된 자기·교차 어텐션, 다음 토큰 손실을 확인한다.
실제 번역 모델이 아닌 단일 층의 소형 교육용 구현이다. 입력은 가상 토큰 번호로 만든 문장이다.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,d=32,heads=4):
        super().__init__()
        self.heads,self.head_dim=heads,d//heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
    def forward(self,query,key_value,causal=False,name="Attention"):
        b,n,d=query.shape
        m=key_value.shape[1]
        q=self.q(query).reshape(b,n,self.heads,self.head_dim).transpose(1,2)
        k=self.k(key_value).reshape(b,m,self.heads,self.head_dim).transpose(1,2)
        v=self.v(key_value).reshape(b,m,self.heads,self.head_dim).transpose(1,2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal:  # 미래 토큰의 키를 참조하지 못하도록 음의 무한대로 가린다.
            mask=torch.triu(torch.ones(n,m,device=scores.device,dtype=torch.bool),1)
            scores=scores.masked_fill(mask,float("-inf"))
        weights=scores.softmax(-1)
        z=(weights@v).transpose(1,2).reshape(b,n,d)
        print(name,"Q",tuple(q.shape),"K/V",tuple(k.shape),"attention",tuple(weights.shape))
        return self.out(z)

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=MultiHeadAttention()
        self.ff=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Linear(64,32))
        self.norm1,self.norm2=nn.LayerNorm(32),nn.LayerNorm(32)
    def forward(self,x):
        x=self.norm1(x+self.attn(x,x,name="Encoder self"))
        return self.norm2(x+self.ff(x))  # 원본 Transformer처럼 잔차 덧셈 후 층 정규화를 적용한다.

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn,self.cross_attn=MultiHeadAttention(),MultiHeadAttention()
        self.ff=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Linear(64,32))
        self.norm1,self.norm2,self.norm3=nn.LayerNorm(32),nn.LayerNorm(32),nn.LayerNorm(32)
    def forward(self,x,memory):
        x=self.norm1(x+self.self_attn(x,x,True,"Decoder masked self"))
        x=self.norm2(x+self.cross_attn(x,memory,name="Decoder cross"))
        return self.norm3(x+self.ff(x))

class MiniTransformer(nn.Module):
    def __init__(self,vocab=20,d=32):
        super().__init__()
        self.src_embed,self.tgt_embed=nn.Embedding(vocab,d),nn.Embedding(vocab,d)
        self.encoder,self.decoder=EncoderBlock(),DecoderBlock()
        self.head=nn.Linear(d,vocab)
        self.d=d
    def embed_position(self,ids,layer):
        x=layer(ids)*math.sqrt(self.d)
        p=torch.arange(ids.shape[1],device=ids.device).float().unsqueeze(1)
        div=torch.exp(torch.arange(0,self.d,2,device=ids.device).float()*
                      (-math.log(10000.0)/self.d))
        pe=torch.zeros(ids.shape[1],self.d,device=ids.device)
        pe[:,0::2],pe[:,1::2]=torch.sin(p*div),torch.cos(p*div)
        return x+pe.unsqueeze(0)  # 원 논문의 고정 사인·코사인 위치 인코딩을 더한다.
    def forward(self,src,tgt_input):
        s=self.embed_position(src,self.src_embed)
        t=self.embed_position(tgt_input,self.tgt_embed)
        print("Source/Target embeddings:",tuple(s.shape),tuple(t.shape))
        memory=self.encoder(s)
        print("Encoder memory:",tuple(memory.shape))
        x=self.decoder(t,memory)
        logits=self.head(x)
        print("Decoder logits:",tuple(logits.shape))
        return logits

def main():
    torch.manual_seed(0)
    device="cuda" if torch.cuda.is_available() else "cpu"
    src=torch.tensor([[4,5,6,2]],device=device)
    target=torch.tensor([[1,7,8,9,2]],device=device)  # 가상 문장: 시작 토큰은 1, 종료 토큰은 2다.
    tgt_input=target[:,:-1]  # 교사 강제 학습: 시작 토큰부터 디코더에 입력한다.
    gt=target[:,1:]          # 입력보다 한 칸 뒤에 오는 다음 토큰 정답
    print("Source:",src.tolist(),"Decoder input:",tgt_input.tolist(),"GT:",gt.tolist())
    model=MiniTransformer().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    optimizer.zero_grad()
    logits=model(src,tgt_input)
    loss=F.cross_entropy(logits.reshape(-1,20),gt.reshape(-1))
    before=model.head.weight.detach().clone()
    loss.backward()
    print("Next-token Loss:",loss.item())
    print("Cross-attention gradient norm:",model.decoder.cross_attn.q.weight.grad.norm().item())
    optimizer.step()
    print("Head max update:",(model.head.weight-before).abs().max().item())
    print("교육용 1-step 학습: 실제 번역 품질을 보여주지 않습니다.")

if __name__=="__main__":
    main()

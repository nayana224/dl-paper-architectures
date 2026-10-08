"""DeepLabV3+ (2018): ASPP multi-scale context + low-level Decoder, Pixel Loss.
교육용 축소판. 논문의 Xception backbone, output stride, atrous separable conv 전체 재현은 아님.
"""
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

class ASPP(nn.Module):
    def __init__(self):
        super().__init__()
        # 서로 다른 receptive field를 위한 병렬 dilation branch.
        self.branches=nn.ModuleList([
            nn.Sequential(nn.Conv2d(32,8,1),nn.ReLU()),
            *[nn.Sequential(nn.Conv2d(32,8,3,padding=r,dilation=r),nn.ReLU())
              for r in (1,2,3)]])
        self.global_branch=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(32,8,1),nn.ReLU())
        self.project=nn.Conv2d(40,32,1)
    def forward(self,x):
        ys=[m(x) for m in self.branches]
        g=self.global_branch(x)
        ys.append(F.interpolate(g,size=x.shape[-2:],mode="nearest"))
        print("ASPP branches:",[tuple(y.shape) for y in ys])
        return self.project(torch.cat(ys,dim=1))

class MiniDeepLabV3Plus(nn.Module):
    def __init__(self):
        super().__init__()
        self.low=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU())
        self.high=nn.Sequential(nn.Conv2d(16,32,3,2,1),nn.ReLU())
        self.aspp=ASPP()
        self.low_proj=nn.Conv2d(16,8,1)
        self.decoder=nn.Sequential(nn.Conv2d(40,24,3,padding=1),nn.ReLU(),
                                   nn.Conv2d(24,24,3,padding=1),nn.ReLU(),
                                   nn.Conv2d(24,2,1))
    def forward(self,x):
        print("Image:",tuple(x.shape))
        low=self.low(x)
        high=self.high(low)
        print("Low/High features:",tuple(low.shape),tuple(high.shape))
        context=self.aspp(high)
        up=F.interpolate(context,size=low.shape[-2:],mode="bilinear",align_corners=False)
        merged=torch.cat([up,self.low_proj(low)],dim=1)
        print("Decoder concat:",tuple(merged.shape))
        logits=self.decoder(merged)
        logits=F.interpolate(logits,size=x.shape[-2:],mode="bilinear",align_corners=False)
        print("Pixel logits:",tuple(logits.shape))
        return logits

def main():
    torch.manual_seed(0)
    device="cuda" if torch.cuda.is_available() else "cpu"
    image=Image.new("RGB",(64,64),"white")
    mask=Image.new("L",(64,64),0)
    ImageDraw.Draw(image).rectangle((14,14,50,50),fill="red")
    ImageDraw.Draw(mask).rectangle((14,14,50,50),fill=1)
    x=TF.to_tensor(image).unsqueeze(0).to(device)
    gt=torch.tensor(list(mask.getdata()),dtype=torch.long,device=device).reshape(1,64,64)
    model=MiniDeepLabV3Plus().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    opt.zero_grad()
    logits=model(x)
    loss=F.cross_entropy(logits,gt)
    before=model.decoder[-1].weight.detach().clone()
    loss.backward()
    print("GT:",tuple(gt.shape),"Loss:",loss.item())
    print("ASPP grad norm:",model.aspp.branches[1][0].weight.grad.norm().item())
    opt.step()
    print("Head max update:",(model.decoder[-1].weight-before).abs().max().item())

if __name__=="__main__":
    main()

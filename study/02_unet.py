"""U-Net (2015): Encoder/Decoder와 skip concat, pixel-level loss.
원 논문은 valid(unpadded) 3x3 conv + crop; 여기서는 same padding과 bilinear upsample로 단순화.
"""
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

class ConvPair(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),
                                    nn.Conv2d(b,b,3,padding=1),nn.ReLU())
    def forward(self,x): return self.layers(x)

class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2 = ConvPair(3,8), ConvPair(8,16)
        self.bridge = ConvPair(16,32)
        self.dec2, self.dec1 = ConvPair(32+16,16), ConvPair(16+8,8)
        self.head = nn.Conv2d(8,2,1)

    def forward(self,x):
        print("Image:",tuple(x.shape))
        e1=self.enc1(x)
        e2=self.enc2(F.max_pool2d(e1,2))
        b=self.bridge(F.max_pool2d(e2,2))
        print("Encoder, Bottleneck:",tuple(e1.shape),tuple(e2.shape),tuple(b.shape))
        up2=F.interpolate(b,size=e2.shape[-2:],mode="bilinear",align_corners=False)
        cat2=torch.cat([up2,e2],dim=1)  # skip: encoder feature 채널 결합
        print("Skip concat 2:",tuple(cat2.shape))
        d2=self.dec2(cat2)
        up1=F.interpolate(d2,size=e1.shape[-2:],mode="bilinear",align_corners=False)
        cat1=torch.cat([up1,e1],dim=1)
        print("Skip concat 1:",tuple(cat1.shape))
        logits=self.head(self.dec1(cat1))
        print("Pixel logits:",tuple(logits.shape))
        return logits

def main():
    torch.manual_seed(0)
    device="cuda" if torch.cuda.is_available() else "cpu"
    image=Image.new("RGB",(64,64),"black")
    mask=Image.new("L",(64,64),0)
    ImageDraw.Draw(image).ellipse((16,16,48,48),fill="green")
    ImageDraw.Draw(mask).ellipse((16,16,48,48),fill=1)
    x=TF.to_tensor(image).unsqueeze(0).to(device)
    gt=torch.tensor(list(mask.getdata()),dtype=torch.long,device=device).reshape(1,64,64)
    model=MiniUNet().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    opt.zero_grad()
    logits=model(x)
    # logits [B,Classes,H,W], GT [B,H,W]; 1 pixel당 class 분류.
    loss=F.cross_entropy(logits,gt)
    before=model.head.weight.detach().clone()
    loss.backward()
    print("GT:",tuple(gt.shape),"Loss:",loss.item(),"Grad norm:",model.head.weight.grad.norm().item())
    opt.step()
    print("Head max weight update:",(model.head.weight-before).abs().max().item())
    print("교육용 1 step이며 의미 있는 segmentation 성능을 나타내지 않습니다.")

if __name__=="__main__":
    main()

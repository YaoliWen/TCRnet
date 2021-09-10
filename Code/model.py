# 
import torch
from torch import nn
import math
# 
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
# 
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual + out
        out = self.relu(out)

        return out
# 
class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0, scale=None):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim = -1)
        self.scale = scale
    def forward(self, q, k, v):
        """
        Args:
            q: Queries[B, n, L_q, D_q]
            k: Keys[B, n, L_k, D_k]
            v: Values[B, n, L_v, D_v]
            scale: 浮点标量
            attn_mask: Masking[B, L_q, L_k]
        """

        attention = torch.matmul(q, k.transpose(2,3))
        if self.scale:
            attention = attention / self.scale
        # if attn_mask is not None:
        #     attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        # where_nan = torch.isnan(attention)
        # attention[where_nan] = 0
        context = torch.matmul(attention, v)
        return context, attention
# 
class MultiHeaAttention(nn.Module):
    def __init__(self, model_dim, num_heads, d_k, d_v, dropout, bias=False):
        super(MultiHeaAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.model_dim = self,model_dim
        
        self.linear_q = nn.Linear(model_dim, d_k*num_heads, bias=False)
        self.linear_k = nn.Linear(model_dim, d_k*num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, d_v*num_heads, bias=False)
        
        self.attn_dot = ScaledDotProductAttention(dropout, scale=d_k ** 0.5)
        self.final_linear = nn.Linear(num_heads*d_v, model_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, key, value, query):
        batch_size = key.size(0) # B
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = query.view(batch_size, -1, self.num_heads, self.d_k) # [B,H*W,n_h,d_q]
        key = key.view(batch_size, -1, self.num_heads, self.d_k) # [B,H*W,n_h,d_k]
        value = value.view(batch_size, -1, self.num_heads, self.d_v) # [B,H*W,n_h,d_v]

        query = query.transpose(1, 2) # [B,n_h,H*W,d_q]
        key = key.transpose(1, 2) # [B,n_h,H*W,d_k]
        value = value.transpose(1, 2) # [B,n_h,H*W,d_v]

        out, attention = self.attn_dot(q=query, k=key, v=value)  # [B,n_h,H*W,d_v]

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_v*self.num_heads) # [B,H*W,d_v*n_h]
        
        out = self.final_linear(out) # [B,H*W,D]
        out = self.dropout(out)
        return out, attention
# 
class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout, bias=True):
        super(EncoderLayer, self).__init__()
        d_k= model_dim//num_heads
        d_v= model_dim//num_heads
        self.mult_attn = MultiHeaAttention(model_dim, num_heads, d_k, d_v, dropout, bias=False)
        self.fc1 = nn.Linear(model_dim, 4*model_dim, bias)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(4*model_dim, model_dim, bias)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
    
    def forward(self, query):
        # query [B,D,H,W]
        batch = query.size(0) # B
        dim = query.size(1) # D
        height = query.size(2) # H
        width = query.size(3) # W
        query = query.permute(0, 2, 3, 1) # [B,H,W,D]
        query = query.view(batch, -1, dim) # [B,H*W,D]
        residual = query
        query = self.layer_norm1(query)
        out, attention = self.mult_attn(key=query, value=query, query=query) # [B,H*W,D]
        
        out = residual + out  # [B,H*W,D]
        residual = out
        out = self.layer_norm2(out)
        out = self.fc1(out)
        out = self.gelu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        out = residual + out # [B,H*W,D]
        out = out.view(batch,height,width,dim) # [B,H,W,D]
        out = out.permute(0, 3, 1, 2) # [B,D,H,W]
        return out, attention
# 
class Vit_EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout, bias=True):
        super(Vit_EncoderLayer, self).__init__()
        d_k= model_dim//num_heads
        d_v= model_dim//num_heads
        self.mult_attn = MultiHeaAttention(model_dim, num_heads, d_k, d_v, dropout, bias=False)
        self.fc1 = nn.Linear(model_dim, 4*model_dim, bias)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(4*model_dim, model_dim, bias)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, model_dim)) # [1,1,D]
    
    def forward(self, query):
        # query [B,D,H,W]
        batch = query.size(0) # B
        dim = query.size(1) # D
        query = query.permute(0, 2, 3, 1) # [B,H,W,D]
        query = query.view(batch, -1, dim) # [B,H*W,D]
        cls_tokens = self.cls.repeat(batch,1,1) # [B,1,D]
        query = torch.cat((cls_tokens, query), dim=1) # [B,1+H*W,D]
        residual = query
        query = self.layer_norm1(query)
        out, attention = self.mult_attn(key=query, value=query, query=query) # [B,1+H*W,D], [B,1+H*W,1+H*W]
        out = residual + out  # [B,1+H*W,D]
        residual = out
        out = self.layer_norm2(out)
        out = self.fc1(out)
        out = self.gelu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        out = residual + out # [B,1+H*W,D]
        out = out[:,0,:] # [B,D]
        return out, attention
# 
class TCRnet(nn.Module):
    def __init__(self, num_classes, trans_layer='0', pool_type='avg', num_heads=8, blocks=2, dropout=0.0, bias=True, model_type=None, is_BN=False):
        super(TCRnet, self).__init__() # 输入 B*3*224*224
        self.inplanes = 64
        self.model_type = model_type
        self.trans_layer = trans_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  #(224-6+6)/2=112 64*112*112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # (112-2+2)/2=56 64*56*56
        self.layer1 = self._make_layer(block=BasicBlock, planes=64, blocks=blocks, stride=1) # 64*56*56->64*56*56
        if '1' in self.trans_layer:
            self.trans1 = EncoderLayer(model_dim=64, num_heads=num_heads, dropout=dropout, bias=bias)
        self.layer2 = self._make_layer(block=BasicBlock, planes=128, blocks=blocks, stride=2) # 128*28*28->128*28*28
        if '2' in self.trans_layer:
            self.trans2 = EncoderLayer(model_dim=128, num_heads=num_heads, dropout=dropout, bias=bias)
        self.layer3 = self._make_layer(block=BasicBlock, planes=256, blocks=blocks, stride=2) # 256*14*14—>256*14*14
        if '3' in self.trans_layer:
            self.trans3 = EncoderLayer(model_dim=256, num_heads=num_heads, dropout=dropout, bias=bias)
        self.layer4 = self._make_layer(block=BasicBlock, planes=512, blocks=blocks, stride=2) # 512*7*7->512*7*7
        if pool_type=='gap':
            self.avgpool = nn.AdaptiveAvgPool2d(1) # 512*7*7->B*512*1*1
        if pool_type=='avg':
            self.trans4 = EncoderLayer(model_dim=512, num_heads=num_heads, dropout=dropout, bias=bias) # 512*7*7->512*7*7
            self.avgpool = nn.AdaptiveAvgPool2d(1) # 512*7*7->B*512*1*1
        if pool_type=='vit':
            self.vitpool = Vit_EncoderLayer(model_dim=512, num_heads=num_heads, dropout=dropout, bias=bias) # 512*7*7->B*512
        self.pool_type = pool_type
        self.softmax = nn.Softmax(-1)
        self.fc = nn.Linear(512, num_classes)
        self.is_BN = is_BN
        if is_BN:
            self.bn2 = nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B,3,224,224] [B,C,224,224]
        attention = []
        # forward
        f = self.conv1(x) # [B, D, H, W]
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f) # [B,64,56,56]
        f = self.layer1(f)  # [B,64,56,56]

        if '1' in self.trans_layer:
            f, attention1 = self.trans1(f) # [B,64,56,56]
            attention.append(attention1) # [B,num_head,3136,3136]

        f = self.layer2(f) # [B,128,28,28]

        if '2' in self.trans_layer:
            f, attention2 = self.trans2(f) # [B,128,28,28]
            attention.append(attention2) # [B,num_head,784,784]

        f = self.layer3(f) # [B,256,14,14]

        if '3' in self.trans_layer:
            f, attention3 = self.trans3(f) # [B,256,14,14]
            attention.append(attention3) # [B,num_head,196,196]

        f = self.layer4(f) # [B,512,7,7]

        if self.pool_type=='gap':
            f = self.avgpool(f) # [B,512,1,1]
            f = f.squeeze(3).squeeze(2) # [B,512]
        if self.pool_type=='avg':
            f, attention4 = self.trans4(f) # [B,512,7,7]
            f = self.avgpool(f) # [B,512,1,1]
            f = f.squeeze(3).squeeze(2) # [B,512]
            attention.append(attention4) # [B,num_head,49,49]
        if self.pool_type=='vit':
            f, attention4 = self.vitpool(f) # [B,512]
            attention.append(attention4)
        
        if self.is_BN:
            f = self.bn2(f)
        prec_score = self.fc(f)
        return prec_score, attention
 

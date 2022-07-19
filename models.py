import torch
import torch.nn as nn
import seaborn
import matplotlib.pyplot as plt
from utils import get_2d_sincos_pos_embed

class PatchEmbed(nn.Module):
    '''
    image to patch
    eg: bs*3*128*128 -> bs*768*8*8 -> bs*64*768
    '''
    def __init__(self, img_size=128, patch_size=16, in_chans=3, 
                 embed_dim=768, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class MultiHeadAttention(nn.Module):
    '''
    multi head attentionblock
    input: bs*N*dim
    qkv:bs*3N*dim -> bs*N*3*h*(dim/h) -> 3 bs*h*N*(dim/h)
    q@k:bs*h*N*N
    a@v:bs*h*N*(dim/h) -> bs*N*dim
    proj:bs*N*dim
    output: bs*N*dim
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = (dim//num_heads) ** -0.5
    
    def forward(self,x):
        B, N, dim = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    '''
    layer scale
    input:bs*N*dim
    every 1*dim multiple a learnable gamma(1*dim)
    output:bs*N*dim
    '''
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_drop=0., 
                 attn_drop=0., init_values=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ls = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x):
        x = x + self.ls(self.attn(self.norm(x)))
        return x

class MAEViT(nn.Module):
    '''
    Masked AutoEncoder with ViT backbone
    '''
    def __init__(self, img_size=128, patch_size=16, in_chans=3,
                 encoder_embed_dim=768, encoder_depth=4, encoder_num_heads=8,
                 decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=8,
                 norm_layer=nn.LayerNorm, maskrate=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed()
        self.num_patch = int((img_size//patch_size)**2)
        self.encoder_pos_embed = nn.Parameter(torch.zeros((1, self.num_patch, encoder_embed_dim)), requires_grad=False)
        self.decoder_pos_embed = nn.Parameter(torch.zeros((1, self.num_patch, decoder_embed_dim)), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.maskrate = maskrate

        self.encoder_blocks = nn.ModuleList([
            ViTBlock(encoder_embed_dim, encoder_num_heads, qkv_bias=True, norm_layer=norm_layer)
            for i in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        self.decoder_embed = nn.Linear(encoder_embed_dim,decoder_embed_dim,bias=True)

        self.decoder_blocks = nn.ModuleList([
            ViTBlock(decoder_embed_dim, decoder_num_heads, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.init_weights()

    def init_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        encoderposembed = get_2d_sincos_pos_embed(self.encoder_pos_embed.shape[-1],int(self.num_patch**0.5))
        self.encoder_pos_embed.data.copy_(torch.from_numpy(encoderposembed).float().unsqueeze(0))
        decoderposembed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],int(self.num_patch**0.5))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoderposembed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

    def patchify(self, imgs):
        '''
        bs*3*h*w -> bs*N*dim
        '''
        bs,c,h,w = imgs.shape
        p = self.patch_size
        h = h//p
        w = w//p
        x = imgs.reshape((bs, c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape((bs, h * w, p**2 * c))
        return x

    def unpatchify(self, x):
        '''
        bs*N*dim -> bs*3*h*w
        '''
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def masking(self,x):
        '''
        bs*N*dim -> bs*(N*maskrate)*dim shuffled
        '''
        bs, N, dim = x.shape
        len_keep = int(N*(1-self.maskrate))

        r = torch.rand(bs,N,device=x.device)
        shuffle_index = torch.argsort(r, dim = 1)
        keep_index = shuffle_index[:,:len_keep]
        x = x.gather(dim=1,index=keep_index.unsqueeze(-1).repeat(1, 1, dim))
        return x, shuffle_index

    def unmasking(self,x,maskid):
        '''
        bs*(N*maskrate)*dim -> bs*N*dim unshuffled 
        '''
        mask_tokens = self.mask_token.repeat(x.shape[0], maskid.shape[1]-x.shape[1], 1)
        x = torch.cat([x,mask_tokens],dim = 1)
        index = torch.argsort(maskid, dim=1)
        x = x.gather(dim=1,index=index.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        return x

    def forward_encoder(self,x):
        x = self.patch_embed(x)
        x += self.encoder_pos_embed
        x, maskid = self.masking(x)
        for b in self.encoder_blocks:
            x = b(x)
        x = self.encoder_norm(x)
        return x, maskid

    def forward_decoder(self,x,maskid):
        x = self.decoder_embed(x)
        x = self.unmasking(x,maskid)
        x += self.decoder_pos_embed
        for b in self.decoder_blocks:
            x = b(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def forward_loss(self,pred,imgs,maskid):
        bs, N, dim = pred.shape
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        mask = torch.ones((bs, N), device=imgs.device)
        mask[:,:int(N*(1-self.maskrate))] = 0
        index = torch.argsort(maskid, dim=1)
        mask = mask.gather(dim=1,index=index)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self,x):
        imgs = x
        x,maskid = self.forward_encoder(x)
        pred = self.forward_decoder(x,maskid)
        loss = self.forward_loss(pred,imgs,maskid)
        return loss,pred,maskid

    def show_pos_embed(self):
        '''
        show encoder position embed
        '''
        seaborn.heatmap(self.encoder_pos_embed.squeeze(0))
        plt.title('encoder position embeding')
        plt.show()
        plt.savefig('posembed.png')

    def show_masks(self, x):
        '''
        input one img 1*3*h*w
        output 2 img : original input with masks and MAE output
        '''
        img = x
        if (img.shape[0] != 1) :
            print('please use only one img input')
        x, maskid = self.forward_encoder(x)
        mask_img = self.patchify(img)
        ori_img = mask_img
        bs, N, dim = mask_img.shape

        len_keep = int(N*(1-self.maskrate))
        keep_index = maskid[:,:len_keep]
        mask_img = mask_img.gather(dim=1,index=keep_index.unsqueeze(-1).repeat(1, 1, dim))
        mask_tokens = nn.Parameter(torch.zeros(1, 1, dim),requires_grad=False).to(x.device)
        mask_tokens = mask_tokens.repeat(1, N-len_keep, 1)
        mask_img = torch.cat([mask_img,mask_tokens],dim = 1)
        index = torch.argsort(maskid, dim=1)
        mask_img = mask_img.gather(dim=1,index=index.unsqueeze(-1).repeat(1, 1, dim))
        mask_img = self.unpatchify(mask_img)

        pred = self.forward_decoder(x,maskid)
        mask = torch.ones((bs, N), device=img.device)
        mask[:,:len_keep] = 0
        index = torch.argsort(maskid, dim=1)
        mask = mask.gather(dim=1,index=index)
        mask = mask.unsqueeze(-1).repeat(1,1,dim)
        pred = pred*mask+ori_img*(1-mask)
        pred_img = self.unpatchify(pred)
        return mask_img,pred_img

#device = 'cuda:0'
#x = torch.ones((1,3,128,128)).to(device)
#MAE = MAEViT(maskrate=0.75)
#MAE.to(device)
#x = MAE(x)
#print(x[1].shape)

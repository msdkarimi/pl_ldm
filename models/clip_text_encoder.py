from utils.build import register_model
import torch.nn as nn
import open_clip
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from utils.diffusion_util import checkpoint

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

# class FrozenCLIPEmbedder(AbstractEncoder):
#     """Uses the CLIP transformer encoder for text (from Hugging Face)"""
#     # def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
#     def __init__(self, version="../pretrained/clip_model", device="cuda", max_length=77):
#         super().__init__()
#         print('clip model loaded from {}'.format(version))
#         self.tokenizer = CLIPTokenizer.from_pretrained(version)
#         self.transformer = CLIPTextModel.from_pretrained(version)
#         self.device = device
#         self.max_length = max_length
#         self.freeze()
#
#     def freeze(self):
#         self.transformer = self.transformer.eval()
#         for param in self.parameters():
#             param.requires_grad = False
#
#     def forward(self, text):
#         batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
#                                         return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
#         tokens = batch_encoding["input_ids"].to(self.device)
#         outputs = self.transformer(input_ids=tokens)
#
#         z = outputs.last_hidden_state
#         return z
#
#     def encode(self, text):
#         return self(text)
#



class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 # freeze=True, layer="penultimate"):
                 freeze=True, layer="penultimate"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)



if __name__ == "__main__":
    from utils.utils import count_params
    # txt_encoder = FrozenCLIPEmbedder().cuda()
    txt_encoder = FrozenOpenCLIPEmbedder().cuda()
    count_params(txt_encoder, verbose=True)
    # for i in range(1):
    _txt = ['hi i am just a test']
    z_txt = txt_encoder.encode(_txt)
    print(z_txt.shape)
    print(z_txt)


@register_model
def clip_constractor(*args, **kwargs):
    return FrozenOpenCLIPEmbedder()



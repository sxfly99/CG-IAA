import os
import timm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import GPT2Tokenizer
import models.clip as clip
from models.gpt2_prefix import load_aesmodel
from models.gpt2_prefix_eval import generate_beam
from models.transformer_mapper import *

# Set proxy for downloading models from Hugging Face
# Note: If you're using a proxy, please replace 7890 with your actual proxy port
# If you don't need a proxy, you can comment out these lines
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

class BaseModel(nn.Module):
    def __init__(self, clip_name):
        super(BaseModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_size = self.select_clip(clip_name)
        self.clip_model = clip_model.float()
        self.clip_size = clip_size['feature_size']
    def select_clip(self, clip_name):
        param = {'feature_size': 512}
        if clip_name == 'RN50':
            clip_model, _ = clip.load("RN50", device=self.device)
            param['feature_size'] = 1024
        elif clip_name == 'ViT-B/16':
            clip_model, _ = clip.load("ViT-B/16", device=self.device)
            param['feature_size'] = 768

        return clip_model, param

    def forward(self, x, texts):
        img_embedding = self.clip_model.visual(x)
        img_embedding = img_embedding @ self.clip_model.visual.proj
        try:
            text_tokens = torch.cat([clip.tokenize(text) for text in texts])
            text_embedding = self.clip_model.encode_text(text_tokens.to(self.device)).float()
            return img_embedding, text_embedding
        except:
            print('Error: ', texts)
            return img_embedding, img_embedding

class AesCritique(nn.Module):
    """
    AesCritique: Multi-Attribute Aesthetic Critique Model
    
    This model generates aesthetic comments across four attributes:
    - Color: Evaluates color harmony, saturation, and visual appeal
    - Composition: Analyzes layout, balance, and structural elements
    - Depth of Field (DoF): Assesses focus, blur effects, and depth perception
    - General: Provides overall aesthetic quality assessment
    
    The model uses CLIP for visual encoding and GPT-2 for text generation,
    with specialized expert models for each aesthetic attribute.
    """
    def __init__(
        self,
        clip_model_path='checkpoints/base_model.pt',
        color_model_path='checkpoints/color.pt',
        composition_model_path='checkpoints/composition.pt',
        dof_model_path='checkpoints/dof.pt',
        general_model_path='checkpoints/general.pt'
    ):
        """
        Args:
            clip_model_path: Path to CLIP base model weights
            color_model_path: Path to color expert model weights
            composition_model_path: Path to composition expert model weights
            dof_model_path: Path to depth of field expert model weights
            general_model_path: Path to general expert model weights
        """
        super(AesCritique, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load base CLIP model
        base_model = BaseModel(clip_name='ViT-B/16')
        weights = torch.load(clip_model_path)
        print('Loading contrast model：', base_model.load_state_dict(weights))
        self.base_model = base_model.clip_model

        # Load tokenizer and expert models
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.prefix_length = 40
        self.expert_color = load_aesmodel(path_weights_mode=color_model_path).cuda()
        self.expert_composition = load_aesmodel(path_weights_mode=composition_model_path).cuda()
        self.expert_df = load_aesmodel(path_weights_mode=dof_model_path).cuda()
        self.expert_general = load_aesmodel(path_weights_mode=general_model_path).cuda()

    def forward(self, x):
        bs = x.size(0)
        # 生成文本描述
        color_comments = []
        composition_comments = []
        df_comments = []
        general_comments = []
        with torch.no_grad():
            prefix = self.base_model.encode_image(x).to(self.device, dtype=torch.float32)
            prefix = prefix @ self.base_model.visual.proj
            prefix = F.normalize(prefix, dim=1)
            # 生成不同角度的prefix

            prefix_color = self.expert_color.clip_project(prefix).reshape(bs, self.prefix_length, -1)
            prefix_compositon = self.expert_composition.clip_project(prefix).reshape(bs, self.prefix_length, -1)
            prefix_df = self.expert_df.clip_project(prefix).reshape(bs, self.prefix_length, -1)
            prefix_general = self.expert_general.clip_project(prefix).reshape(bs, self.prefix_length, -1)

            for prefix_embed_i in prefix_color:
                generated_text_prefix = \
                generate_beam(self.expert_color, self.tokenizer, embed=prefix_embed_i.unsqueeze(0))[0]
                color_comments.append(generated_text_prefix.rstrip())
            for prefix_embed_i in prefix_compositon:
                generated_text_prefix = \
                generate_beam(self.expert_composition, self.tokenizer, embed=prefix_embed_i.unsqueeze(0))[0]
                composition_comments.append(generated_text_prefix.rstrip())
            for prefix_embed_i in prefix_df:
                generated_text_prefix = \
                generate_beam(self.expert_df, self.tokenizer, embed=prefix_embed_i.unsqueeze(0))[0]
                df_comments.append(generated_text_prefix.rstrip())
            for prefix_embed_i in prefix_general:
                generated_text_prefix = \
                generate_beam(self.expert_general, self.tokenizer, embed=prefix_embed_i.unsqueeze(0))[0]
                general_comments.append(generated_text_prefix.rstrip())

        return color_comments, composition_comments, df_comments, general_comments
from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel

from ....utils import dist_utils


class ImageTextConditionedMixin:
    """
    Mixin for image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, *args, image_cond_model: str = 'dinov2_vitl14_reg', text_cond_model: str = 'openai/clip-vit-large-patch14', **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.text_cond_model_name = text_cond_model
        self.image_cond_model = None     # the model is init lazily
        self.text_cond_model = None
        
    @classmethod
    def prepare_for_training(cls, image_cond_model: str, text_cond_model: str = 'openai/clip-vit-large-patch14', **kwargs):
        """
        Prepare for training.
        """
        parent = super()
        if hasattr(parent, 'prepare_for_training'):
            parent.prepare_for_training(**kwargs)
        # download the model
        torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        # pre-download text encoder/tokenizer to cache
        with dist_utils.local_master_first():
            CLIPTextModel.from_pretrained(text_cond_model)
            AutoTokenizer.from_pretrained(text_cond_model)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            # 修复路径格式问题
            # 使用正确的torch.hub.load格式
            dinov2_model = torch.hub.load('/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main', self.image_cond_model_name, source='local', pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }

    def _init_text_cond_model(self):
        """
        Initialize the text conditioning model.
        """
        # load model
        with dist_utils.local_master_first():
            model = CLIPTextModel.from_pretrained(self.text_cond_model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.text_cond_model_name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])
        
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
            # ensure float and range [0,1]
            if image.dtype != torch.float32:
                image = image.float()
            if image.max() > 1.0:
                image = image / 255.0
            # resize to 518 if needed
            if image.shape[-1] != 518 or image.shape[-2] != 518:
                image = F.interpolate(image, size=(518, 518), mode='bilinear', align_corners=False)
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model['transform'](image).cuda()
        # use forward_features to obtain patch tokens before final norm
        features = self.image_cond_model['model'].forward_features(image)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
    
    def get_cond(self, img_cond=None, txt_cond=None, **kwargs):
        if txt_cond == None:
            assert txt_cond is not None, "txt_cond is None from dataloader"
        img_tokens = self.encode_image(img_cond) if img_cond is not None else None
        txt_tokens = self.encode_text(txt_cond) if txt_cond is not None else None
        kwargs['neg_img_cond'] = torch.zeros_like(img_tokens)
        kwargs['neg_txt_cond'] = self.text_cond_model['null_cond'].repeat(txt_tokens.shape[0], 1, 1)
        cond = super().get_mixin_cond(img_tokens, txt_tokens, **kwargs)
        return cond

    
    def get_inference_cond(self, img_cond, txt_cond, **kwargs):
        """
        推理阶段：本方法内完成编码并返回 {'img_cond','txt_cond','neg_cond'}。
        注意：不向上调用 super()，避免 neg_cond 在多处重复传递。
        """
        img_tokens = self.encode_image(img_cond)
        txt_tokens = self.encode_text(txt_cond) if isinstance(txt_cond, list) else txt_cond
        if txt_tokens is None:
            raise ValueError("get_inference_cond requires txt_cond as list[str] or encoded tensor")
        neg_img_tokens = torch.zeros_like(img_tokens)
        neg_txt_tokens = self.text_cond_model['null_cond'].repeat(txt_tokens.shape[0], 1, 1)
        return {'img_cond': img_tokens, 'txt_cond': txt_tokens, 'neg_img_cond': neg_img_tokens, 'neg_txt_cond': neg_txt_tokens}


    def vis_cond(self, img_cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {'image': {'value': img_cond, 'type': 'image'}}

        
    @torch.no_grad()
    def encode_text(self, text) -> torch.Tensor:
        """
        Encode the text.
        """
        # 处理None或空值
        if text is None:
            return None
            
        # 确保text是列表格式
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            text = [str(text)]
            
        # 确保列表中的元素都是字符串
        text = [str(t) if not isinstance(t, str) else t for t in text]
        
        if self.text_cond_model is None:
            self._init_text_cond_model()
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        attn_mask = encoding['attention_mask'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens, attention_mask=attn_mask).last_hidden_state
        # 确保返回float32类型，避免与模型的dtype不匹配
        return embeddings.float()




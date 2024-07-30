from typing import Optional, Union, Tuple, Dict, Any
import torch
from transformers import (LlavaForConditionalGeneration,
                          AutoProcessor,
                          AutoModelForCausalLM,
                          LogitsProcessor,
                          PreTrainedModel,
                          LogitsProcessorList,
                          ProcessorMixin)
from math import log

class CDProcessorWithLM(LogitsProcessor):

    def __init__(self,
                 language_model,
                 processor,
                 prompt_tokens,
                 prompt_attention_mask,
                 arch='llava',
                 alpha=2.0,
                 beta=0.1,
                 entropy=True):
        assert arch in ('enc_dec', 'llava', 'decoder')
        self.alpha = alpha
        self.beta = beta
        self.language_model = language_model
        self.output_so_far = None
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.prompt_tokens= prompt_tokens
        self.arch = arch
        self.prompt_attention_mask = prompt_attention_mask
        self.entropy = entropy
        self.processor = processor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        orig_device = input_ids.device

        if self.arch == 'enc_dec':
            inputs = {
                'input_ids': self.prompt_tokens.to(self.language_model.device),
                'attention_mask': self.prompt_attention_mask.to(self.language_model.device),
                'decoder_input_ids': input_ids.to(self.language_model.device)
            }
        else:
            if self.arch == 'llava':
                new_input_ids = input_ids.clone()
                bs = new_input_ids.shape[0]
                new_input_ids = new_input_ids.flatten()
                new_input_ids = new_input_ids[new_input_ids != 32000]
                new_input_ids = new_input_ids.reshape(bs, -1)
                new_att_mask = torch.ones_like(new_input_ids)
            elif self.arch == 'decoder':
                new_input_ids = torch.cat([self.prompt_tokens.to(self.language_model.device), input_ids.to(self.language_model.device)], dim=-1)
                new_att_mask = torch.cat([self.prompt_attention_mask.to(self.language_model.device), torch.ones_like(input_ids)], dim=-1)
            else:
                raise ValueError('Unsupported architecture, use "llava", "enc_dec" or "decoder"')
            inputs = {
                'input_ids': new_input_ids,
                'attention_mask': new_att_mask
            }

        scores = torch.log_softmax(scores, dim=-1)

        without_image_logits = self.language_model(**inputs).logits[:, -1, :].to(orig_device)

        without_image_logits = torch.log_softmax(without_image_logits, dim=-1)

        def calculate_entropy(logitz):
            p = torch.exp(logitz)
            ent = -(logitz * p).nansum(-1, keepdim=True)
            return ent

        if self.entropy:
            lm_ent = calculate_entropy(without_image_logits)
            alpha = self.alpha / lm_ent
        else:
            alpha = self.alpha

        if scores.shape != without_image_logits.shape:
            # in llava
            dim1, dim2 = scores.shape[-1], without_image_logits.shape[-1]
            assert dim1 > dim2
            without_image_logits = torch.cat([without_image_logits, torch.zeros(scores.shape[0], dim1 - dim2).to(without_image_logits.device)], dim=-1)

        cutoff = log(self.beta) + scores.max(dim=-1, keepdim=True).values
        diffs = (1.0 + alpha) * scores - alpha * without_image_logits
        cd_logits = diffs.masked_fill(scores < cutoff, -float('inf'))

        if torch.isnan(cd_logits).any():
            cd_logits = torch.nan_to_num(cd_logits, nan=float('-inf'))

        cd_logits = torch.log_softmax(cd_logits, dim=-1)

        return cd_logits

class CDModelDelegator:
    """
    A delegator wrapper for transformer models that automatically applies CDProcessorWithLM
    during the generate method call.
    """
    def __init__(self, model: PreTrainedModel,
                 processor: ProcessorMixin,
                 lm_model: PreTrainedModel,
                 alpha: float = 2.0,
                 beta: float = 0.1,
                 entropy: bool = True,
                 arch: str = 'llava'):
        self.model = model
        self.processor = processor
        self.lm_model = lm_model
        self.alpha = alpha
        self.beta = beta
        self.entropy = entropy
        self.arch = arch

    def __getattr__(self, name: str):
        return getattr(self.model, name)

    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            *args,
            **kwargs
    ) -> Union[torch.LongTensor, Tuple[torch.LongTensor, torch.FloatTensor]]:

        if isinstance(inputs, dict):
            prompt_input_ids = inputs.get('input_ids')
            prompt_attention_mask = inputs.get('attention_mask')
        elif isinstance(inputs, torch.Tensor):
            prompt_input_ids = inputs
            prompt_attention_mask = torch.ones_like(inputs)
        else:
            assert 'input_ids' in kwargs
            prompt_input_ids = kwargs['input_ids']
            if 'attention_mask' in kwargs:
                prompt_attention_mask = kwargs['attention_mask']
            else:
                prompt_attention_mask = torch.ones_like(prompt_input_ids)

        # Create CDProcessorWithLM
        cd_processor = CDProcessorWithLM(
            language_model=self.lm_model,
            processor=self.processor,
            prompt_tokens=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            alpha=self.alpha,
            beta=self.beta,
            entropy=self.entropy,
            arch=self.arch
        )

        # Add CDProcessorWithLM to logits_processor list
        if 'logits_processor' not in kwargs:
            kwargs['logits_processor'] = LogitsProcessorList()
        kwargs['logits_processor'].append(cd_processor)

        return self.model.generate(inputs, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def create_cd_llava_model(cls,
                              llava_model_id: str = "llava-hf/llava-1.5-7b-hf",
                              vicuna_model_id: str = "lmsys/vicuna-7b-v1.5",
                              cache_dir: str = None,
                              device: str = 'cuda',
                              torch_dtype: torch.dtype = torch.float16,
                              alpha: float = 2.0,
                              beta: float = 0.1,
                              entropy: bool = True,
                              llava_model_kwargs: Optional[Dict[str, Any]] = None,
                              vicuna_model_kwargs: Optional[Dict[str, Any]] = None
                              ) -> 'CDModelDelegator':
        llava_model_kwargs = llava_model_kwargs or {}
        vicuna_model_kwargs = vicuna_model_kwargs or {}

        llava_model = LlavaForConditionalGeneration.from_pretrained(
            llava_model_id,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            **llava_model_kwargs
        ).to(device)
        llava_processor = AutoProcessor.from_pretrained(llava_model_id)

        vicuna_model = AutoModelForCausalLM.from_pretrained(
            vicuna_model_id,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            **vicuna_model_kwargs
        ).to(device)

        return cls(
            model=llava_model,
            processor=llava_processor,
            lm_model=vicuna_model,
            alpha=alpha,
            beta=beta,
            entropy=entropy,
            arch='llava'
        )
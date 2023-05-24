import logging, os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForPreTraining
from transformers.modeling_outputs import BaseModelOutput
from model_copyutils import CopyBartWithReg, CopyBart, PureCopyBart, AMRRoberta, AMRBart
from prefix_gen_bart import PrefixGenBartForConditionalGeneration
from projector import Projector
import ipdb

logger = logging.getLogger(__name__)

class GenerativeModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        if self.config.model_type=='degree':
            self.model = Degree(config, tokenizer)
        elif self.config.model_type=='degree+copy':
            self.model = DegreeCopyReg(config, tokenizer)
        elif self.config.model_type=='degree+copy-reg':
            self.model = DegreeCopy(config, tokenizer)
        elif self.config.model_type=='degree+purecopy':
            self.model = DegreePureCopy(config, tokenizer)
        elif self.config.model_type=='AMR+prefixgen':
            self.model = AMRPrefixGen(config, tokenizer)
        elif self.config.model_type=='AMR+prefixgen+copy':
            self.model = AMRPrefixGenCopyReg(config, tokenizer)
        elif self.config.model_type=='AMR+prefixgen+copy-reg':
            self.model = AMRPrefixGenCopy(config, tokenizer)
        elif self.config.model_type=='AMR+prefixgen+purecopy':
            self.model = AMRPrefixGenPureCopy(config, tokenizer)
        else:
            raise ValueError("Model type {} does not support yet.".format(self.config.model_type))

    def forward(self, batch):
        return self.model(batch)
        
    def predict(self, batch, num_beams=4, max_length=50):
        return self.model.predict(batch, num_beams, max_length)
    
    def generate(self, input_ids, attention_mask, num_beams=4, max_length=50, 
                **kwargs):
        self.eval()
        with torch.no_grad():
            output = self.model.generate(input_ids, attention_mask, num_beams, max_length,  **kwargs)
        self.eval()
        return output

    def save_model(self, save_path):
        """
        This save model is created mainly in case we need partial save and load. Such as cases with pretraining.
        """
        self.model.save_model(save_path)

    def load_model(self, load_path):
        """
        This load model is created mainly in case we need partial save and load. Such as cases with pretraining.
        """
        self.model.load_model(load_path)

class Degree(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        logger.info(f'Using model {self.__class__.__name__}')
        logger.info(f'Loading pre-trained model {config.model_name}')
        self.model_config =  AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
        self.model = AutoModelForPreTraining.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, batch):
        outputs = self.model(input_ids=batch.enc_idxs, 
                             attention_mask=batch.enc_attn, 
                             decoder_input_ids=batch.dec_idxs, 
                             decoder_attention_mask=batch.dec_attn, 
                             labels=batch.lbl_idxs, 
                             return_dict=True)
        
        loss = outputs['loss']
        
        return loss
        
    def predict(self, batch, num_beams=4, max_length=50):
        return self.generate(batch.enc_idxs, batch.enc_attn, num_beams, max_length)
    
    def generate(self, input_ids, attention_mask, num_beams=4, max_length=50, **kwargs):
        self.eval()
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          num_beams=num_beams, 
                                          max_length=max_length)
        final_output = []
        for bid in range(len(input_ids)):
            # if self.config.model_name.startswith('google/t5') or self.config.model_name.startswith('t5'):
            #     output_sentence = t5_decode(self.tokenizer, outputs[bid])
            # else:
            #     output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()
        return final_output
    
    def save_model(self, save_path):
        self.model.save_pretrained(save_path)

    def load_model(self, load_path):
        self.model.from_pretrained(load_path)

class DegreeCopyReg(Degree):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        logger.info(f'Using model {self.__class__.__name__}')
        logger.info(f'Loading pre-trained model {config.model_name}')
        if config.model_name.startswith('facebook/bart-'):
            self.model_config =  AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
            self.model_config.output_attentions = True
            self.model_config.use_encoder_prefix = False
            self.model_config.use_cross_prefix = False
            self.model_config.use_decoder_prefix = False
            self.model = CopyBartWithReg.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)

        else:
            raise ValueError("PreTrainModel {} does not support yet.".format(config.model_name))
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def generate(self, input_ids, attention_mask, num_beams=4, max_length=50, **kwargs):
        self.eval()
        with torch.no_grad():
            if num_beams == 1:
                self.model._cache_input_ids = input_ids
            else:
                expanded_return_idx = (
                    torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(input_ids.device)
                )
                self.model._cache_input_ids = input_ids.index_select(0, expanded_return_idx)
                
            outputs = self.model.generate(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          num_beams=num_beams, 
                                          max_length=max_length)
        final_output = []
        for bid in range(len(input_ids)):
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()
        return final_output

class DegreeCopy(DegreeCopyReg):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)

        # overwrite model
        if config.model_name.startswith('facebook/bart-'):
            self.model_config =  AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
            self.model_config.output_attentions = True
            self.model_config.use_encoder_prefix = False
            self.model_config.use_cross_prefix = False
            self.model_config.use_decoder_prefix = False
            self.model = CopyBart.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)

        else:
            raise ValueError("PreTrainModel {} does not support yet.".format(config.model_name))
        self.model.resize_token_embeddings(len(self.tokenizer))

class DegreePureCopy(DegreeCopyReg):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)

        # overwrite model
        if config.model_name.startswith('facebook/bart-'):
            self.model_config =  AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
            self.model_config.output_attentions = True
            self.model_config.use_encoder_prefix = False
            self.model_config.use_cross_prefix = False
            self.model_config.use_decoder_prefix = False
            self.model = PureCopyBart.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)
            
        else:
            raise ValueError("PreTrainModel {} does not support yet.".format(config.model_name))
        self.model.resize_token_embeddings(len(self.tokenizer))
        

class AMRPrefixGenBase(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        """
        Need to init by class
        """

    def get_AMR_embedding(self, AMR_string):
        return self.AMR_model.get_encoder_output(AMR_string)

    def get_prefix(self, AMR_embedding):
        batch_size = AMR_embedding[0].size()[0]
        input_tokens = torch.arange(self.config.prefix_length).long().cuda()
        input_tokens = input_tokens.unsqueeze(0).expand(batch_size, -1)
        input_embeds = self.wte(input_tokens) # bz, prefix_len, dim
        prefix = {}
        if self.model_config.use_encoder_prefix:
            prefix['encoder_prefix'] = self.enc_prefix_projector.project(AMR_embedding, input_embeds)
        if self.model_config.use_cross_prefix:
            prefix['cross_prefix'] = self.cross_prefix_projector.project(AMR_embedding, input_embeds)
        if self.model_config.use_decoder_prefix:
            prefix['decoder_prefix'] = self.dec_prefix_projector.project(AMR_embedding, input_embeds)
        return prefix

    def forward(self, batch):
        AMR_embedding = self.get_AMR_embedding(batch.amrgraph)
        prefix = self.get_prefix(AMR_embedding)
        outputs = self.model(input_ids=batch.enc_idxs,
                             prefix=prefix,
                             attention_mask=batch.enc_attn, 
                             decoder_input_ids=batch.dec_idxs, 
                             decoder_attention_mask=batch.dec_attn, 
                             labels=batch.lbl_idxs, 
                             return_dict=True)
        
        loss = outputs['loss']
        
        return loss
        
    def predict(self, batch, num_beams=4, max_length=50):
        return self.generate(batch.enc_idxs, batch.enc_attn, num_beams, max_length,
        amrgraph=batch.amrgraph)
    
    def generate(self, input_ids, attention_mask, num_beams=4, max_length=50, 
                **kwargs):
        self.eval()
        with torch.no_grad():
            AMR_embedding = self.get_AMR_embedding(kwargs['amrgraph'])
            batch_prefix = self.get_prefix(AMR_embedding)
            if num_beams == 1:
                self.model._cache_input_ids = input_ids
                prefix = batch_prefix
            else:
                expanded_return_idx = (
                    torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(input_ids.device)
                )
                self.model._cache_input_ids = input_ids.index_select(0, expanded_return_idx)
                
                prefix = {}
                for name, each_prefix in batch_prefix.items():
                    prefix[name] = (each_prefix[0].index_select(0, expanded_return_idx), 
                                   each_prefix[1].index_select(0, expanded_return_idx))

            model_kwargs = {'prefix': prefix,
                            'encoder_prefix': prefix['encoder_prefix'] if 'encoder_prefix' in prefix.keys() else None}
            outputs = self.model.generate(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          num_beams=num_beams, 
                                          max_length=max_length,
                                          **model_kwargs)
        final_output = []
        for bid in range(len(input_ids)):
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()
        return final_output
    
    def save_model(self, save_path):
        self.model.save_pretrained(os.path.join(save_path, "checkpoint-bart"))
        torch.save(self.wte.state_dict(), os.path.join(save_path, "wte.mdl"))
        torch.save(self.AMR_model.state_dict(), os.path.join(save_path, "amrmodel.mdl"))
        if self.model_config.use_encoder_prefix:
            self.enc_prefix_projector.save(os.path.join(save_path, "enc_prefix_projector.mdl"))
        if self.model_config.use_cross_prefix:
            self.cross_prefix_projector.save(os.path.join(save_path, "cross_prefix_projector.mdl"))
        if self.model_config.use_decoder_prefix:
            self.dec_prefix_projector.save(os.path.join(save_path, "dec_prefix_projector.mdl"))
    
    def load_model(self, load_path):
        logger.info(f"Loading model from {load_path}")
        self.model.from_pretrained(os.path.join(load_path, "checkpoint-bart"))
        self.wte.load_state_dict(torch.load(os.path.join(load_path, "wte.mdl"), map_location=f'cuda:{self.config.gpu_device}'))
        self.AMR_model.load_state_dict(torch.load(os.path.join(load_path, "amrmodel.mdl"), map_location=f'cuda:{self.config.gpu_device}'))
        if self.model_config.use_encoder_prefix:
            self.enc_prefix_projector.load(os.path.join(load_path, "enc_prefix_projector.mdl"))
        if self.model_config.use_cross_prefix:
            self.cross_prefix_projector.load(os.path.join(load_path, "cross_prefix_projector.mdl"))
        if self.model_config.use_decoder_prefix:
            self.dec_prefix_projector.load(os.path.join(load_path, "dec_prefix_projector.mdl"))

class AMRPrefixGen(AMRPrefixGenBase):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        logger.info(f'Using model {self.__class__.__name__}')
        logger.info(f'Loading pre-trained model {config.model_name}')

        if config.model_name.startswith('facebook/bart-'):
            # main model
            self.model_config =  AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
            self.model_config.output_attentions = True
            self.model_config.use_encoder_prefix = config.use_encoder_prefix
            self.model_config.use_cross_prefix = config.use_cross_prefix
            self.model_config.use_decoder_prefix = config.use_decoder_prefix
            self.model_config.prefix_length = config.prefix_length
            
            self.model = PrefixGenBartForConditionalGeneration.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)
            
            ## Load AMR
            if config.AMR_model_path.startswith('xfbai/AMRBART'):
                self.AMR_model = AMRBart(self.config)
            elif config.AMR_model_path.startswith('roberta'):
                self.AMR_model = AMRRoberta(self.config)
        
            ## Prefix Generator
            self.wte = nn.Embedding(config.prefix_length, config.latent_dim)
            if self.model_config.use_encoder_prefix:
                self.enc_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
            if self.model_config.use_cross_prefix:
                self.cross_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
            if self.model_config.use_decoder_prefix:
                self.dec_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
        else:
            raise ValueError("Model does not support yet.")
        self.model.resize_token_embeddings(len(self.tokenizer))

        if self.config.pretrained_model_path is not None:
            self.load_model(self.config.pretrained_model_path)

        if config.freeze_AMR:
            for param in self.AMR_model.parameters():
                param.requires_grad=False

        if config.freeze_bart:
            for param in self.model.parameters():
                param.requires_grad=False

class AMRPrefixGenCopy(AMRPrefixGenBase):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        logger.info(f'Using model {self.__class__.__name__}')
        logger.info(f'Loading pre-trained model {config.model_name}')

        if config.model_name.startswith('facebook/bart-'):
            # main model
            self.model_config = AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
            self.model_config.output_attentions = True
            self.model_config.use_encoder_prefix = config.use_encoder_prefix
            self.model_config.use_cross_prefix = config.use_cross_prefix
            self.model_config.use_decoder_prefix = config.use_decoder_prefix
            self.model_config.prefix_length = config.prefix_length
            
            self.model = CopyBart.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)
            
            ## Load AMR
            if config.AMR_model_path.startswith('xfbai/AMRBART'):
                self.AMR_model = AMRBart(self.config)
            elif config.AMR_model_path.startswith('roberta'):
                self.AMR_model = AMRRoberta(self.config)
        
            ## Prefix Generator
            self.wte = nn.Embedding(config.prefix_length, config.latent_dim)
            if self.model_config.use_encoder_prefix:
                self.enc_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
            if self.model_config.use_cross_prefix:
                self.cross_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
            if self.model_config.use_decoder_prefix:
                self.dec_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
        else:
            raise ValueError("Model does not support yet.")
        self.model.resize_token_embeddings(len(self.tokenizer))

        if self.config.pretrained_model_path is not None:
            self.load_model(self.config.pretrained_model_path)

        if config.freeze_AMR:
            for param in self.AMR_model.parameters():
                param.requires_grad=False

        if config.freeze_bart:
            for param in self.model.parameters():
                param.requires_grad=False

class AMRPrefixGenCopyReg(AMRPrefixGenBase):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        logger.info(f'Using model {self.__class__.__name__}')
        logger.info(f'Loading pre-trained model {config.model_name}')

        if config.model_name.startswith('facebook/bart-'):
            # main model
            self.model_config = AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
            self.model_config.output_attentions = True
            self.model_config.use_encoder_prefix = config.use_encoder_prefix
            self.model_config.use_cross_prefix = config.use_cross_prefix
            self.model_config.use_decoder_prefix = config.use_decoder_prefix
            self.model_config.prefix_length = config.prefix_length
            
            self.model = CopyBartWithReg.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)
            
            ## Load AMR
            if config.AMR_model_path.startswith('xfbai/AMRBART'):
                self.AMR_model = AMRBart(self.config)
            elif config.AMR_model_path.startswith('roberta'):
                self.AMR_model = AMRRoberta(self.config)
        
            ## Prefix Generator
            self.wte = nn.Embedding(config.prefix_length, config.latent_dim)

            if self.model_config.use_encoder_prefix:
                self.enc_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
            if self.model_config.use_cross_prefix:
                self.cross_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
            if self.model_config.use_decoder_prefix:
                self.dec_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
        else:
            raise ValueError("Model does not support yet.")
        self.model.resize_token_embeddings(len(self.tokenizer))

        if self.config.pretrained_model_path is not None:
            self.load_model(self.config.pretrained_model_path)

        if config.freeze_AMR:
            for param in self.AMR_model.parameters():
                param.requires_grad=False

        if config.freeze_bart:
            for param in self.model.parameters():
                param.requires_grad=False

class AMRPrefixGenPureCopy(AMRPrefixGenBase):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        logger.info(f'Using model {self.__class__.__name__}')
        logger.info(f'Loading pre-trained model {config.model_name}')

        if config.model_name.startswith('facebook/bart-'):
            # main model
            self.model_config = AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
            self.model_config.output_attentions = True
            self.model_config.use_encoder_prefix = config.use_encoder_prefix
            self.model_config.use_cross_prefix = config.use_cross_prefix
            self.model_config.use_decoder_prefix = config.use_decoder_prefix
            self.model_config.prefix_length = config.prefix_length
            
            self.model = PureCopyBart.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)
            
            ## Load AMR
            if config.AMR_model_path.startswith('xfbai/AMRBART'):
                self.AMR_model = AMRBart(self.config)
            elif config.AMR_model_path.startswith('roberta'):
                self.AMR_model = AMRRoberta(self.config)
        
            ## Prefix Generator
            self.wte = nn.Embedding(config.prefix_length, config.latent_dim)

            if self.model_config.use_encoder_prefix:
                self.enc_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
            if self.model_config.use_cross_prefix:
                self.cross_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
            if self.model_config.use_decoder_prefix:
                self.dec_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
        else:
            raise ValueError("Model does not support yet.")
        self.model.resize_token_embeddings(len(self.tokenizer))

        if self.config.pretrained_model_path is not None:
            self.load_model(self.config.pretrained_model_path)

        if config.freeze_AMR:
            for param in self.AMR_model.parameters():
                param.requires_grad=False

        if config.freeze_bart:
            for param in self.model.parameters():
                param.requires_grad=False
        

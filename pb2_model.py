import math
import collections

# import timm
import torch
# import argparse
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
# from torch.utils.data import Dataset, DataLoader

import loralib as lora
# Define if Debugging
LORA_DEBUG = False
import torch
import torch.nn.functional as F

class DiverseBeamSearch:
    def __init__(self, num_groups=4, diversity_lambda=0.5, beam_size=5):
        """
        Args:
            num_groups: Number of diverse groups
            diversity_lambda: Diversity strength penalty
            beam_size: Beam size per group
        """
        self.num_groups = num_groups
        self.diversity_lambda = diversity_lambda
        self.beam_size = beam_size
        self.group_size = beam_size // num_groups
        assert beam_size % num_groups == 0, "Beam size must be divisible by num_groups"

    def compute_diversity_penalty(self, previous_group_tokens, logits):
        """
        Compute diversity penalty based on previous group tokens
        Args:
            previous_group_tokens: List of tokens selected by previous groups
            logits: Current prediction logits (batch_size, vocab_size)
        Returns:
            penalty: Tensor of shape (batch_size, vocab_size)
        """
        batch_size, seq_len, vocab_size = logits.size()
        penalty = torch.zeros((batch_size, seq_len, vocab_size), device=logits.device)
        
        if not previous_group_tokens:
            return penalty
            
        for tokens in previous_group_tokens:
            for token in tokens:
                # Make sure token is within vocabulary size
                if token < vocab_size:
                    penalty[:, :, token] += 1.0
        
        return penalty

    def diverse_beam_step(self, decoder, image_feat, text_id, current_step):
        """
        Perform one step of diverse beam search
        Args:
            decoder: Decoder model
            image_feat: Image features [batch_size, 257, 768]
            text_id: Input text tokens [batch_size, beam_size, seq_len]
            current_step: Current generation step
        """
        batch_size = text_id.size(0)
        beam_size = text_id.size(1)
        seq_len = text_id.size(2)
        
        # Reshape image_feat to match beam size
        # image_feat = image_feat.unsqueeze(1).expand(-1, beam_size, -1, -1)  # [batch_size, beam_size, 257, 768]
        # image_feat = 
        # Reshape for decoder
        # image_feat_reshaped = image_feat.contiguous().view(batch_size * beam_size, -1, image_feat.size(-1))
        text_id_reshaped = text_id.contiguous().view(batch_size * beam_size, -1)
        
        all_scores = []
        all_tokens = []
        previous_group_tokens = []

        # Generate predictions for each group
        for group_id in range(self.group_size):
            # Get decoder outputs
            decode_with_img = decoder(image_feat, text_id_reshaped, mode = "beam")  # [batch_size * beam_size, seq_len, vocab_size]
            print("decode_with_img size:", decode_with_img.size())
            logits = decode_with_img[:, -1, :]  # Take last token predictions [batch_size * beam_size, vocab_size]
            print("logits size:", logits.size())
            # Reshape logits back to include beam dimension
            logits = logits.view(batch_size, beam_size, -1)
            
            # Apply diversity penalty
            diversity_penalty = self.compute_diversity_penalty(previous_group_tokens, logits)
            logits = logits - self.diversity_lambda * diversity_penalty
            
            # Get top-k scores and tokens for this group
            scores = F.log_softmax(logits, dim=-1)  # [batch_size, beam_size, vocab_size]
            print("scores size:", scores.size())
            top_scores, top_tokens = scores.topk(
                self.group_size, 
                dim=-1,
                largest=True,
                sorted=True
            )
            print("top_scores size:", top_scores.size())
            all_scores.append(top_scores)
            all_tokens.append(top_tokens)
            previous_group_tokens.append(top_tokens.view(-1).tolist())
        # print("all_scores size:", all_scores.size())
        # Combine results from all groups
        step_scores = torch.cat(all_scores, dim=1)  # [batch_size, beam_size]
        step_tokens = torch.cat(all_tokens, dim=1)  # [batch_size, beam_size]
        
        return step_scores, step_tokens

    def diverse_beam_search(self, encoder, decoder, image, max_length=100):
        """
        Full diverse beam search implementation
        """
        batch_size = image.size(0)
        image_feat = encoder.forward_features(image)  # [batch_size, 257, 768]
        
        # Initialize with start tokens
        text_id = torch.ones((batch_size, self.beam_size, 1), 
                            dtype=torch.long, 
                            device=image.device)  # [batch_size, beam_size, 1]
        
        # Track sequences and scores
        sequences = text_id
        sequence_scores = torch.zeros((batch_size, self.beam_size), device=image.device)
        
        # Generate sequences
        for step in range(max_length):
            # Get predictions for this step
            step_scores, step_tokens = self.diverse_beam_step(
                decoder,
                image_feat,
                sequences,
                step
            )
            
            # Update sequence scores
            print("step_scores size:", step_scores.size())
            print("sequence_scores size:", sequence_scores.size())
            sequence_scores += step_scores
            
            # Add new tokens to sequences
            step_tokens = step_tokens.unsqueeze(-1)  # Add sequence dimension
            sequences = torch.cat([sequences, step_tokens], dim=2)
            
            # Check for completion (e.g., EOS token)
            if (step_tokens == self.eos_token_id).any():
                break

        # Apply length penalty
        sequence_lengths = (sequences != self.eos_token_id).sum(dim=-1).float()
        length_penalty = ((5 + sequence_lengths) / 6.0) ** 0.6
        final_scores = sequence_scores / length_penalty

        # Get best sequences
        best_scores, best_indices = final_scores.max(dim=-1)
        best_sequences = torch.gather(
            sequences, 
            1, 
            best_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, 1, sequences.size(-1)
            )
        ).squeeze(1)

        return best_sequences, best_scores

    # In your model class:
    def autoreg_infer_beam_v2(self, image, beam=4, step=20):
        """
        Wrapper function for diverse beam search
        """
        diverse_beam_search = DiverseBeamSearch(
            num_groups=2,
            diversity_lambda=0.5,
            beam_size=beam
        )
        
        sequences, scores = diverse_beam_search.diverse_beam_search(
            self.encoder,
            self.decoder,
            image,
            max_length=step
        )
        return sequences

    def sample_diverse(self, logits):
        """
        Sample diverse tokens using nucleus sampling within each group
        """
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus_mask = cumsum_probs < 0.9  # Use top 90% probability mass
        sorted_probs[~nucleus_mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled_indices = torch.multinomial(sorted_probs, num_samples=1)
        sampled_tokens = torch.gather(sorted_indices, -1, sampled_indices)
        return sampled_tokens

diverse_beam_search = DiverseBeamSearch(
    num_groups=2,          # Number of diverse groups
    diversity_lambda=0.5,  # Diversity penalty strength
    beam_size=4          # Total beam size (must be divisible by num_groups)
)
class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

Rank = 56
alpha = 8
class Attention_Lora(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lora_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=Rank, lora_alpha=alpha)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=Rank, lora_alpha=alpha)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding (1,60,768)
        q, k, v  = self.lora_attn(x).split(self.n_embd, dim=2)
        # k size: (B, self.n_head, T, C // self.n_head)
        # q size: (B, self.n_head, T, C // self.n_head)
        # v size: (B, self.n_head, T, C // self.n_head)
        # att size: (B, self.n_head, T, T)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # By applying softmax along dim=-1
        # , we ensure that each query's attention weights are normalized over the keys.
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))
# class Cross_Attention(nn.Module):

#     def __init__(self, cfg):
#         super().__init__()
#         self.c_attn = nn.Linear(cfg.n_embd, cfg.n_embd)
#         self.i_attn = nn.Linear(cfg.n_embd, 2*cfg.n_embd)
#         self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
#         self.n_head = cfg.n_head
#         self.n_embd = cfg.n_embd
#         size = cfg.block_size
#         self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

#     def forward(self, x, encoder_output):
#         # batch, context, embedding
#         B, T, C = x.size() # (b,max_len,768)
#         q = self.c_attn(x) # (b,max_len,768)
#         q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (b,12,max_len,64)
        
#         B, I, C = encoder_output.size()              # (b,60,768)
#         k, v = self.i_attn(encoder_output).split(self.n_embd, dim=2)    # (b,257,768)
#         k = k.view(B, I, self.n_head, C // self.n_head).transpose(1, 2) # (b,12,257,64)
#         v = v.view(B, I, self.n_head, C // self.n_head).transpose(1, 2)
#         # q (1,12,257,64), kT (1,12,64,60)
#         # q@kT = att (1,12,257,60)
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         att = F.softmax(att, dim=-1)  #(1,12, 257,257)
#         # att@v.T = (1,257,12,64)
#         return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C)) #(1,257,768)
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention_Lora(cfg)
        # self.crossattn = Cross_Attention(cfg)  # Cross Attention
        # multi-layer perceptron
        self.mlp = nn.Sequential(
            collections.OrderedDict(
                [
                    ("c_fc", lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=Rank, lora_alpha=alpha)),
                    ("act", nn.GELU(approximate="tanh")),
                    ("c_proj", lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=Rank, lora_alpha=alpha)),
                ]
            )
        )

    def forward(self, x):
        # print("x size:", x.size()) # (batch, max_len, embedding)
        # print("encoder_output size:", encoder_output.size()) #torch.Size([4, 197, 768])
        # print("self.ln_1(x) size:", self.ln_1(x).size()) #torch.Size([4, 100, 768])
        # concatenate x and encoder_output
        # x = torch.cat((x, encoder_output), dim=1) # (batch, max_len+197, embedding)
        # print("x concat size:", x.size())
        x = x + self.attn(self.ln_1(x))
        # x = x + self.crossattn(x, encoder_output)
        x = x + self.mlp(self.ln_2(x))
        return x
class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            # wte = lora.Embedding(cfg.vocab_size, cfg.n_embd, r=Rank),
            # wpe = lora.Embedding(cfg.block_size, cfg.n_embd, r=Rank),
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        # self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, bias=False, r=Rank)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # self.linear = nn.Linear(1408, cfg.n_embd)  # (1,257,1408->768)
        # self.linear = nn.Linear(1280, cfg.n_embd)  
        # self.linear = nn.Linear(1024, cfg.n_embd)
        self.linear = nn.Linear(1664, cfg.n_embd)

        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, encoder_output, caption_in, mode="None"):
        # print("caption_in size:", caption_in.size()) # (batch, max_len)
        encoder_output = self.linear(encoder_output)
        x = torch.narrow(caption_in, 1, 0, min(caption_in.size(1), self.block_size))    #(1,max_cap_len)
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0) #(1,max_cap_len)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)             #(1,max_cap_len,768)
        # print("x size:", x.size()) # (batch, max_len, embedding) 
        # print("encoder_output size:", encoder_output.size()) #torch.Size([batch, 257, 768])
        # concat x with encoder_output
        # print
        if mode == "beam":
            # x = x.squeeze(0)
            # duplicate encoder_output to match x
            encoder_output = encoder_output.repeat(x.size(0), 1, 1)
        x = torch.cat((encoder_output, x), dim=1)

        # print("x concat size:", x.size())
        for layer in self.transformer.h:
            x = layer(x)
        # x (1,257,768)
        x = self.lm_head(self.transformer.ln_f(x)) # x (1,max_cap_len,50257)
        # print("x size after lm_head:", x.size())
        return x
    def forward_decode(self, x: Tensor, encoder_output: Tensor):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        # x = self.lm_head(self.transformer.ln_f(self.transformer.h(x)))
        attn_probs = 0
        for block in self.transformer.h:
            x0, attn_probs = block(x, encoder_output)  # Pass both x and encoder_output to each block
            x = x0

        x = self.lm_head(self.transformer.ln_f(x))
        # x = self.lm_head(self.transformer.ln_f(self.transformer.h(x, encoder_output)))
        return x, attn_probs
BOS_TOKEN = 50256
EOS_TOKEN = 50256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Transformer_lora(nn.Module):
    def __init__(self, encoder, decoder_weights):
        super().__init__()
        self.encoder = encoder
        decoder_cfg = Config(checkpoint=decoder_weights)
        self.decoder = Decoder(decoder_cfg)
    def forward(self, images, instruction):
        if LORA_DEBUG:
            print("images size:", images.size())
            print("instruction size:", instruction.size())
        features = self.encoder.forward_features(images)
        if LORA_DEBUG:
            print("features size:", features.size())
            print("self.decoder size:", self.decoder(features, instruction).size())
        return self.decoder(features, instruction)
    
    def beam_search(self, img, beams=3, max_length=30):
        self.eval()
        print("img size:", img.size())
        if img.dim() < 4:
            img = img.unsqueeze(0)
        # print("img size:", img.size())
        def forward_prob(x: Tensor, encoder_feature: Tensor):
            x = torch.narrow(x, 1, 0, min(x.size(1), self.decoder.block_size))
            pos = torch.arange(
                x.size()[1], dtype=torch.long, device=x.device
            ).unsqueeze(0)
            x = self.decoder.transformer.wte(x) + self.decoder.transformer.wpe(pos)
            x = torch.cat((encoder_feature, x), dim=1)
            for block in self.decoder.transformer.h:
                x = block(x)
            # Generator
            # 根據seq的最後一個字分類
            x = self.decoder.lm_head(self.decoder.transformer.ln_f(x[:, -1, :]))
            return x

        encoder_feature = self.encoder.forward_features(img)
        encoder_feature = self.decoder.linear(encoder_feature)
        id_list = []
        prpb_list = []

        # ----------/Beam Searching/-----------#
        cur_state = torch.tensor([EOS_TOKEN]).to(device).unsqueeze(1)
        n_probs = forward_prob(cur_state, encoder_feature)

        vocab_size = n_probs.shape[-1]

        # 選擇概率最高的beams個單詞作為初始候選序列
        # probs, pred id
        cur_probs, n_chars = n_probs.log_softmax(-1).topk(k=beams, axis=-1)
        cur_probs = cur_probs.reshape(beams)
        n_chars = n_chars.reshape(beams, 1)
        # gen first k beams
        cur_state = cur_state.repeat((beams, 1))  # 複製 beams 次
        cur_state = torch.cat((cur_state, n_chars), axis=1)

        for i in range(max_length - 1):
            rm_set = set()
            # to get top k
            n_probs = forward_prob(
                cur_state, encoder_feature.repeat((beams, 1, 1))
            ).log_softmax(-1)

            cur_probs = (
                cur_probs.unsqueeze(-1) + n_probs
            ).flatten()  # (beams*vocab) 攤平成1D

            # length normalization
            _, idx = (cur_probs / (len(cur_state[0]) + 1)).topk(k=beams, dim=-1)
            cur_probs = cur_probs[idx]

            # to generate next char
            n_chars = (torch.remainder(idx, vocab_size)).unsqueeze(-1)

            # 找回屬於哪個beam
            top_k = (idx / vocab_size).long()
            cur_state = torch.cat((cur_state[top_k], n_chars), dim=1)

            # concat next_char to beams
            for idx, char in enumerate(n_chars):
                if char.item() == EOS_TOKEN or cur_state.size(1) == max_length:
                    id_list.append(cur_state[idx].cpu().tolist())
                    prpb_list.append(cur_probs[idx].item() / len(id_list[-1]))
                    rm_set.add(idx)
                    beams -= 1

            to_keep_idx = [i for i in range(len(cur_state)) if i not in rm_set]
            if len(to_keep_idx) == 0:
                break
            cur_state = cur_state[to_keep_idx]
            cur_probs = cur_probs[to_keep_idx]

        max_idx = torch.argmax(torch.tensor(prpb_list)).item()

        # 把50256抽離
        id_list[max_idx] = [x for x in id_list[max_idx] if x != EOS_TOKEN]
        return id_list[max_idx]
    
    def autoreg_infer_beam(self, image, beam, step = 100):
        # Implement autogressive inference with beam search here
        image_feat = self.encoder.forward_features(image)
        image_emb = self.decoder.linear(image_feat) # batch x (patches + 1) x 768 (patches = 196, 256...)
        batch_cnt = image_emb.size(0)
        text_id = torch.ones((batch_cnt, 1, 1), dtype=torch.long, device= "cuda") * 50256 # batch x context(1) x path
        text_prob = torch.ones((batch_cnt, 1), dtype=torch.float) # batch x path
        
        # print("batch_cnt:", batch_cnt)
        for i in range(step):
            text_pred_all = torch.zeros((batch_cnt, 50257, 0))
            path_cnt = text_id.size(2)
            # print("path_cnt:", path_cnt)
            for j in range(path_cnt):
                # decode = self.decoder(image_feat,text_id[:, :, j])
                
                # print("decode size:", decode.size())
                # print("emb size:", image_emb.size())
                decode_with_img = self.decoder(image_feat,text_id[:, :, j])
                # remove image embedding
                decode = decode_with_img[:, 257:, :]
                text_pred_path = F.softmax(decode, dim=-1).detach().cpu() # batch x 1 x 50257
                
                text_pred_all = torch.cat((text_pred_all, text_pred_path[:, i, :].unsqueeze(2)), dim=-1)
            text_pred_topk = torch.topk(text_pred_all, beam, dim=1)
            text_pred_id = text_pred_topk.indices # batch x beam x path
            text_pred_scores = text_pred_topk.values # batch x beam x path

            text_prob_prev = text_prob.unsqueeze(1).repeat(1, beam, 1) # batch x beam x path
            text_pred_scores = text_prob_prev * text_pred_scores * 2 # batch x beam x path

            new_text_id = text_id.repeat(1, 1, beam).detach().cpu() # batch x context x beam*path
            new_text_pred_id = text_pred_id.view(batch_cnt, 1, beam*path_cnt) # batch x 1 x beam*path
            new_text_pred_scores = text_pred_scores.view(batch_cnt, 1, beam*path_cnt) # batch x 1 x beam*path
            final = torch.cat((new_text_id, new_text_pred_id), dim=1) # batch x context+2 x beam*path
            
            # Sort by probability
            text_prob = torch.zeros((batch_cnt, beam*path_cnt), dtype=torch.float)
            for k in range(batch_cnt):
                sort_score, indexes = new_text_pred_scores[k].squeeze().sort(descending=True)
                final[k] = final[k][:, indexes]
                text_prob[k] = sort_score
            text_id = final.to("cuda")
            text_id = text_id[:, :, :beam]
            text_prob = text_prob[:, :beam]

        return text_id[:,:,0]
    def autoreg_infer_beam_v2(self, image, beam, step=100, early_stop_threshold=0.95, end_token_id=2):  # Added parameters
        # Implement autoregressive inference with beam search here
        image_feat = self.encoder.forward_features(image)
        image_emb = self.decoder.linear(image_feat) # batch x (patches + 1) x 768 (patches = 196, 256...)
        batch_cnt = image_emb.size(0)
        text_id = torch.ones((batch_cnt, 1, 1), dtype=torch.long, device="cuda") * 50256 # batch x context(1) x path
        text_prob = torch.ones((batch_cnt, 1), dtype=torch.float) # batch x path
        
        # Track completed sequences and their scores
        completed_sequences = []
        completed_scores = []
        
        for i in range(step):
            text_pred_all = torch.zeros((batch_cnt, 50257, 0))
            path_cnt = text_id.size(2)
            
            for j in range(path_cnt):
                decode_with_img = self.decoder(image_feat,text_id[:, :, j])
                # remove image embedding
                decode = decode_with_img[:, 257:, :]
                text_pred_path = F.softmax(decode, dim=-1).detach().cpu() # batch x 1 x 50257
                
                text_pred_all = torch.cat((text_pred_all, text_pred_path[:, i, :].unsqueeze(2)), dim=-1)
            
            text_pred_topk = torch.topk(text_pred_all, beam, dim=1)
            text_pred_id = text_pred_topk.indices # batch x beam x path
            text_pred_scores = text_pred_topk.values # batch x beam x path
            
            text_prob_prev = text_prob.unsqueeze(1).repeat(1, beam, 1) # batch x beam x path
            text_pred_scores = text_prob_prev * text_pred_scores * 2 # batch x beam x path
            
            new_text_id = text_id.repeat(1, 1, beam).detach().cpu() # batch x context x beam*path
            new_text_pred_id = text_pred_id.view(batch_cnt, 1, beam*path_cnt) # batch x 1 x beam*path
            new_text_pred_scores = text_pred_scores.view(batch_cnt, 1, beam*path_cnt) # batch x 1 x beam*path
            final = torch.cat((new_text_id, new_text_pred_id), dim=1) # batch x context+2 x beam*path
            
            # Sort by probability
            text_prob = torch.zeros((batch_cnt, beam*path_cnt), dtype=torch.float)
            for k in range(batch_cnt):
                # Check for completed sequences (ending with end_token_id)
                for b in range(beam):
                    if new_text_pred_id[k, 0, b] == end_token_id:
                        sequence = final[k, :, b].cpu()
                        score = new_text_pred_scores[k, 0, b].item()
                        completed_sequences.append(sequence)
                        completed_scores.append(score)
                        
                        # Early stopping condition - if we find a very good candidate
                        if score > early_stop_threshold:
                            return sequence.unsqueeze(0)
                
                sort_score, indexes = new_text_pred_scores[k].squeeze().sort(descending=True)
                final[k] = final[k][:, indexes]
                text_prob[k] = sort_score
            
            # If all beams have completed sequences, return the best one
            if len(completed_sequences) >= beam:
                best_idx = completed_scores.index(max(completed_scores))
                return completed_sequences[best_idx].unsqueeze(0)
            
            text_id = final.to("cuda")
            text_id = text_id[:, :, :beam]
            text_prob = text_prob[:, :beam]
        
        # If we reach here without early stopping, return the most probable sequence
        return text_id[:,:,0]
        
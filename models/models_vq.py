import torch
import torch.nn.functional as F
import importlib
from einops import rearrange
from torch.nn import Embedding
# Removed discriminator import for inference
# from models.discriminator import NLayerDiscriminator, weights_init
from models.lpips import LPIPS
from models.encoder_decoder import Encoder, Decoder, Decoder_Cross

    
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class VQModel(torch.nn.Module):
    def __init__(self,
                 args,
                 ddconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.args = args
        
        self.stage = args.stage
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # Removed discriminator for inference
        
        embed_dim = args.embed_dim
        # Removed perceptual loss for inference
        # self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = args.rate_p        
        self.quantize_type = args.quantizer_type

        print("****Using Quantizer: %s"%(args.quantizer_type))
        self.criterion = torch.nn.CrossEntropyLoss()
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        codebook_dim = embed_dim
        if args.tuning_codebook == -1: ## Random
            print("****Using Tuned Random Codebook****")
            print("Word Number:%d" %(args.n_vision_words))
            print("Feature Dim:%d" %(embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, embed_dim)
            self.tok_embeddings.weight.data.uniform_(-1.0 / args.n_vision_words, 1.0 / args.n_vision_words)
            self.tok_embeddings.weight.requires_grad = True
        
        elif args.tuning_codebook == -2: ##Random Fix
            print("****Using Fix Random Codebook****")
            print("Word Number:%d" %(args.n_vision_words))
            print("Feature Dim:%d" %(embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, embed_dim)
            self.tok_embeddings.weight.data.uniform_(-1.0 / args.n_vision_words, 1.0 / args.n_vision_words)
            self.tok_embeddings.weight.requires_grad = False

        elif args.tuning_codebook == 0:
            print("****Using Fix Initialized Codebook****")
            checkpoint = torch.load(args.local_embedding_path, map_location="cpu")
            args.n_vision_words = checkpoint.shape[0]
            codebook_dim = checkpoint.shape[1]
            print("Word Number:%d" %(args.n_vision_words))
            print("Feature Dim:%d" %(embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, checkpoint.shape[1])
            self.tok_embeddings.weight.data = checkpoint
            self.tok_embeddings.weight.data = self.tok_embeddings.weight.data.float()
            self.tok_embeddings.weight.requires_grad = False

        elif args.tuning_codebook == 1:
            print("****Tuning Initialized Codebook****")
            checkpoint = torch.load(args.local_embedding_path, map_location="cpu")
            args.n_vision_words = checkpoint.shape[0]
            codebook_dim = checkpoint.shape[1]
            print("Word Number:%d" %(args.n_vision_words))
            print("Feature Dim:%d" %(embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, checkpoint.shape[1])
            self.tok_embeddings.weight.data = checkpoint
            self.tok_embeddings.weight.data = self.tok_embeddings.weight.data.float()
            self.tok_embeddings.weight.requires_grad = True

        self.e_dim = embed_dim
        self.remap = remap
        self.sane_index_shape = sane_index_shape
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if args.use_cblinear == 1:
            print("****Using Linear Codebook Projector****")
            self.codebook_projection = torch.nn.Linear(codebook_dim, embed_dim)
            torch.nn.init.normal_(self.codebook_projection.weight, std=embed_dim ** -0.5)
        elif args.use_cblinear == 2:
            print("****Using MLP Codebook Projector****")
            self.codebook_projection = torch.nn.Sequential(
                torch.nn.Linear(codebook_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, embed_dim),
            )
            #torch.nn.init.normal_(self.codebook_projection.weight, std=embed_dim ** -0.5)

        if self.quantize_type == "ema":
            self.decay = 0.99
            self.eps = 1e-5
            self.cluster_size = torch.nn.Parameter(torch.zeros(args.n_vision_words), requires_grad = False)
            self.embed_avg = torch.nn.Parameter(self.tok_embeddings.weight.clone(), requires_grad = False)
            self.update = True
            self.tok_embeddings.weight.requires_grad = False
            self.num_tokens = args.n_vision_words
        elif self.quantize_type == "norm_vq":
            print("****Using NormVQ Quantizer****")
            # 使用candidate_ratio替代固定num_candidates
            candidate_ratio = getattr(args, 'candidate_ratio', 0.1)
            num_candidates = int(args.n_vision_words * candidate_ratio)
            self.num_candidates = max(1, min(num_candidates, args.n_vision_words))
            print(f"Candidate ratio: {candidate_ratio}, Num candidates: {self.num_candidates}")
            # 为norm_vq算法准备缓存变量
            self.register_buffer('sorted_norms_sq', None, persistent=False)
            self.register_buffer('sorted_indices', None, persistent=False)
            self.register_buffer('codebook_norms_sq', None, persistent=False)
            self.register_buffer('offsets', torch.arange(self.num_candidates, dtype=torch.long).unsqueeze(0), persistent=False)
            self._is_cached = False



    def _cache_sorted_embeddings(self, tok_embeddings_weight):
        codebook_norms_sq = tok_embeddings_weight.square().sum(dim=1)
        sorted_norms_sq, sorted_indices = torch.sort(codebook_norms_sq)

        self.codebook_norms_sq = codebook_norms_sq
        self.sorted_norms_sq = sorted_norms_sq
        self.sorted_indices = sorted_indices
        self._is_cached = True

    @torch.no_grad()
    def _get_candidates(self, input_norms_sq, tok_embeddings_weight):
        if self.training:
            codebook_norms_sq = tok_embeddings_weight.square().sum(dim=1)
            sorted_norms_sq, sorted_indices = torch.sort(codebook_norms_sq)
        else:
            if not self._is_cached:
                self._cache_sorted_embeddings(tok_embeddings_weight)
            codebook_norms_sq = self.codebook_norms_sq
            sorted_norms_sq = self.sorted_norms_sq
            sorted_indices = self.sorted_indices

        search_idx = torch.searchsorted(sorted_norms_sq, input_norms_sq.squeeze(-1), right=False)
        half_k = self.num_candidates // 2
        max_start = self.tok_embeddings.num_embeddings - self.num_candidates
        start_indices = (search_idx - half_k).clamp(0, max_start)
        candidate_positions = start_indices.unsqueeze(1) + self.offsets
        candidate_indices = sorted_indices[candidate_positions]
        candidate_norms_sq = codebook_norms_sq[candidate_indices]
        return candidate_indices, candidate_norms_sq 


    def quantize(self, z, temp=None, rescale_logits=False, return_logits=False):

        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # 获取码本权重（考虑投影）
        if self.args.use_cblinear != 0:
            tok_embeddings_weight = self.codebook_projection(self.tok_embeddings.weight)
        else:
            tok_embeddings_weight = self.tok_embeddings.weight

        if self.quantize_type == "norm_vq":
            # 新的NormVQ算法实现 - 高效版本
            input_norms_sq = z_flattened.detach().square().sum(dim=1, keepdim=True)
            
            # 获取候选码本索引
            candidate_indices, candidate_norms_sq = self._get_candidates(input_norms_sq,tok_embeddings_weight)
            
            # 1. 找出所有候选索引中的唯一值，并建立映射关系
            unique_indices, inverse_indices = torch.unique(candidate_indices, return_inverse=True)
            
            # 2. 创建临时的、小的子码本 (sub-codebook)
            unique_cand_vecs = F.embedding(unique_indices, tok_embeddings_weight.detach())
            
            # 3. 高效计算点积
            all_dots = torch.matmul(z_flattened.detach(), unique_cand_vecs.T)
            
            # 4. 使用 inverse_indices 来 "还原" 每个 input 对应的候选点积
            remapped_indices = inverse_indices.reshape(candidate_indices.shape)
            dot = torch.gather(all_dots, 1, remapped_indices)
            
            # 5. 计算欧氏距离平方
            dists_sq = input_norms_sq + candidate_norms_sq - 2.0 * dot
            
            
            # 6. 找到最小距离的索引
            min_idx_in_cand = torch.argmin(dists_sq, dim=1)
            min_encoding_indices = candidate_indices.gather(1, min_idx_in_cand.unsqueeze(1)).squeeze(1)
            
            # 7. 获取量化结果
            z_q = F.embedding(min_encoding_indices, tok_embeddings_weight).view(z.shape)
            loss = torch.mean((z_q.detach()-z)**2) + 0.33 * torch.mean((z_q - z.detach()) ** 2)
            
            min_encodings = None
            perplexity = None
            
            # 返回unique_cand_vecs的大小信息
            unique_cand_vecs_info = {
                'size': unique_cand_vecs.shape[0],
                'unique_indices_count': unique_indices.shape[0]
            }
            
        else:
            # 原始距离计算
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(tok_embeddings_weight**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, rearrange(tok_embeddings_weight, 'n d -> d n'))

            min_encoding_indices = torch.argmin(d, dim=1)
            
            if self.quantize_type == "ema":
                z_q = self.tok_embeddings(min_encoding_indices).view(z.shape)
                # For inference, skip EMA updates
                min_encodings = None
                perplexity = None
                loss = F.mse_loss(z_q.detach(), z) 
            else:
                min_encodings = None
                perplexity = None
                z_q = F.embedding(min_encoding_indices, tok_embeddings_weight).view(z.shape)
                loss = torch.mean((z_q.detach()-z)**2) + 0.33 * torch.mean((z_q - z.detach()) ** 2)
    
            # 其他量化类型不提供unique_cand_vecs信息
            unique_cand_vecs_info = None

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (None, min_encodings, min_encoding_indices, unique_cand_vecs_info)
    
    def forward(self, input, global_input=None, data_iter_step=None, step=None, is_val=False):
        
        #encoder_feature = self.quant_conv(self.encoder(input))
        quant, qloss, info = self.encode(input)
        # 解包info元组，正确处理可能包含unique_cand_vecs_info的情况
        if len(info) == 4:
            _, _, tk_labels, _ = info
        else:
            _, _, tk_labels = info

        ###Training GPT
        if self.stage == 2: 
            return quant, tk_labels.view(input.shape[0], -1)
        
        dec = self.decode(quant)
        
        # For inference, only return the reconstructed image
        return dec


    def encode(self, input):
        #print(self.encoder(input))
        h = self.quant_conv(self.encoder(input))
        if self.e_dim == 768 and self.args.tuning_codebook != -1:
            h = h / h.norm(dim=1, keepdim=True)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant, global_c_features=None):
        quant = self.post_quant_conv(quant)

        dec = self.decoder(quant)

        return dec
    
    def decode_code(self, code_b):
        # 获取码本权重（考虑投影）
        if self.args.use_cblinear != 0:
            tok_embeddings_weight = self.codebook_projection(self.tok_embeddings.weight)
        else:
            tok_embeddings_weight = self.tok_embeddings.weight
        quant_b = F.embedding(code_b, tok_embeddings_weight)
        dec = self.decode(quant_b)
        return dec

    def train(self, mode: bool = True):
        if self.quantize_type == "norm_vq":
            if not mode and self.training:
                # 进入eval模式，缓存排序后的码本
                if self.args.use_cblinear != 0:
                    tok_embeddings_weight = self.codebook_projection(self.tok_embeddings.weight)
                else:
                    tok_embeddings_weight = self.tok_embeddings.weight
                self._cache_sorted_embeddings(tok_embeddings_weight)
            if mode and not self.training:
                # 进入train模式，清除缓存
                self._is_cached = False
        return super().train(mode)

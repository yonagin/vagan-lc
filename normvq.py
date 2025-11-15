class NormVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, candidate_ratio: float = 0.1):
        super().__init__()
        assert num_embeddings > 0 and embedding_dim > 0, "num_embeddings/embedding_dim must be > 0"
        assert 0.0 < candidate_ratio <= 1.0, "candidate_ratio must be between 0.0 and 1.0"

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        num_candidates = int(self.num_embeddings * candidate_ratio)
        
        self.num_candidates = max(1, min(num_candidates, self.num_embeddings))

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

        self.register_buffer('sorted_norms_sq', None, persistent=False)
        self.register_buffer('sorted_indices', None, persistent=False)
        self.register_buffer('codebook_norms_sq', None, persistent=False)
        self.register_buffer('offsets', torch.arange(self.num_candidates, dtype=torch.long).unsqueeze(0), persistent=False)

        self._is_cached = False

    def _cache_sorted_embeddings(self):
        
        codebook = self.embedding.weight.detach()
        codebook_norms_sq = codebook.square().sum(dim=1)
        sorted_norms_sq, sorted_indices = torch.sort(codebook_norms_sq)

        self.codebook_norms_sq = codebook_norms_sq
        self.sorted_norms_sq = sorted_norms_sq
        self.sorted_indices = sorted_indices
        self._is_cached = True

    @torch.no_grad()
    def _get_candidates(self, input_norms_sq: torch.Tensor):
        
        if self.training:
            codebook = self.embedding.weight
            codebook_norms_sq = codebook.square().sum(dim=1)
            sorted_norms_sq, sorted_indices = torch.sort(codebook_norms_sq)
        else:
            if not self._is_cached:
                self._cache_sorted_embeddings()
            codebook_norms_sq = self.codebook_norms_sq
            sorted_norms_sq = self.sorted_norms_sq
            sorted_indices = self.sorted_indices

        search_idx = torch.searchsorted(sorted_norms_sq, input_norms_sq.squeeze(-1), right=False)
        half_k = self.num_candidates // 2
        max_start = self.num_embeddings - self.num_candidates
        start_indices = (search_idx - half_k).clamp(0, max_start)
        candidate_positions = start_indices.unsqueeze(1) + self.offsets
        candidate_indices = sorted_indices[candidate_positions]
        candidate_norms_sq = codebook_norms_sq[candidate_indices]
        return candidate_indices, candidate_norms_sq

    def train(self, mode: bool = True):
        
        if not mode and self.training:
            self._cache_sorted_embeddings()
        if mode and not self.training:
            self._is_cached = False
        return super().train(mode)

    def forward(self, inputs: torch.Tensor):
        B, C, H, W = inputs.shape
        inputs_ = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs_.reshape(-1, self.embedding_dim)
        input_norms_sq = flat_input.detach().square().sum(dim=1, keepdim=True)

        # 1. 获取候选索引和它们的范数 (与之前相同)
        # candidate_indices: [N, K], N = B*H*W, K = num_candidates
        # candidate_norms_sq: [N, K]
        candidate_indices, candidate_norms_sq = self._get_candidates(input_norms_sq)

        # 2. 找出所有候选索引中的唯一值，并建立映射关系
        # unique_indices: [U], 包含了所有候选中的不重复索引, U <= N*K
        # inverse_indices: [N*K], 原始展平后的 candidate_indices 中每个元素在 unique_indices 中的位置
        unique_indices, inverse_indices = torch.unique(candidate_indices, return_inverse=True)

        # 3. 创建临时的、小的子码本 (sub-codebook)
        # 只对唯一的索引进行 embedding 操作，避免创建巨大的张量
        # unique_cand_vecs: [U, D], D = embedding_dim
        unique_cand_vecs = F.embedding(unique_indices, self.embedding.weight.detach())

        # 4. 高效计算点积
        # 计算 flat_input 和子码本中所有向量的点积
        # all_dots: [N, U]
        all_dots = torch.matmul(flat_input.detach(), unique_cand_vecs.T)

        # 5. 使用 inverse_indices (映射关系) 来 "还原" 每个 input 对应的候选点积
        # 将 inverse_indices 变回 [N, K] 的形状，其值是 0 到 U-1 的索引
        remapped_indices = inverse_indices.reshape(candidate_indices.shape) # Shape: [N, K]
        # 从 all_dots [N, U] 中为每个输入向量 [N] 挑选出它对应的 K 个候选的点积
        # dot: [N, K]
        dot = torch.gather(all_dots, 1, remapped_indices)
        
        # 6. 计算欧氏距离平方 (与之前相同)
        # input_norms_sq: [N, 1], candidate_norms_sq: [N, K], dot: [N, K]
        dists_sq = input_norms_sq + candidate_norms_sq - 2.0 * dot

        # 7. 找到每个输入向量的最近邻在 K 个候选中的索引
        # min_idx_in_cand: [N]
        min_idx_in_cand = torch.argmin(dists_sq, dim=1)

        # 8. 使用这个局部索引从原始的 candidate_indices 中找到全局码本的最终索引
        # encoding_indices: [N]
        encoding_indices = candidate_indices.gather(1, min_idx_in_cand.unsqueeze(1)).squeeze(1)

        quantized_flat = self.embedding(encoding_indices)
        quantized = quantized_flat.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2).contiguous()
        indices = encoding_indices.view(B, H, W)

        return quantized, indices, None
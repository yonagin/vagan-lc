class NormVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, num_candidates: int = 32):
        super().__init__()
        assert num_embeddings > 0 and embedding_dim > 0, "num_embeddings/embedding_dim must be > 0"

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_candidates = max(1, min(num_candidates, num_embeddings))

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

        candidate_indices, candidate_norms_sq = self._get_candidates(input_norms_sq)

        cand_vecs = F.embedding(candidate_indices, self.embedding.weight.detach())
        dot = torch.einsum('nkd,nd->nk', cand_vecs, flat_input.detach())
        dists_sq = input_norms_sq + candidate_norms_sq - 2.0 * dot

        min_idx_in_cand = torch.argmin(dists_sq, dim=1)
        encoding_indices = candidate_indices.gather(1, min_idx_in_cand.unsqueeze(1)).squeeze(1)

        quantized_flat = self.embedding(encoding_indices)
        quantized = quantized_flat.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2).contiguous()
        indices = encoding_indices.view(B, H, W)

        return quantized, indices
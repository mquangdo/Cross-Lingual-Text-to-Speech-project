import torch

def regotron_monotonic_alignment_loss(
    attention_weights: torch.Tensor,
    text_lens: torch.Tensor,
    mel_lens: torch.Tensor,
    delta: float = 0.01,
) -> torch.Tensor:
    """
    Monotonic Alignment Loss từ paper "Regotron" (Georgiou et al., 2022).

    Công thức:
        h[a_j] = sum_{i=1}^{N} a_ij * i          (centroid tại mel frame j)
        L_A = sum_{j=1}^{M-1} max( h[a_j] - h[a_{j+1}] + delta*(N/M)*N , 0 )

    Ý nghĩa: phạt khi centroid của frame j lớn hơn centroid frame j+1
    (tức là attention đang đi ngược chiều - không monotonic).

    Args:
        attention_weights: [B, T_dec, T_enc] — phải là shape này
        text_lens:         [B] — độ dài thực của encoder (N)
        mel_lens:          [B] — độ dài thực của decoder (M)
        delta:             hyperparameter kiểm soát mức độ phạt (paper dùng 0.01)

    Returns:
        scalar loss
    """
    device = attention_weights.device
    dtype = attention_weights.dtype
    B, T_dec, T_enc = attention_weights.shape

    total_loss = torch.zeros(1, device=device, dtype=dtype)
    valid_count = 0

    for b in range(B):
        N = int(text_lens[b].item())   # số characters
        M = int(mel_lens[b].item())    # số mel frames

        if N <= 0 or M <= 1:
            continue

        # attention của sample b trong vùng hợp lệ: [M, N]
        A = attention_weights[b, :M, :N]  # [M, N]

        # Tính centroid cho mỗi mel frame j:
        # h[a_j] = sum_{i=1}^{N} a_ij * i
        # Dùng index 1-based như trong paper
        char_indices = torch.arange(1, N + 1, device=device, dtype=dtype)  # [N]

        # A: [M, N], char_indices: [N] → centroids: [M]
        centroids = (A * char_indices.unsqueeze(0)).sum(dim=1)  # [M]

        # Tính penalty: max(h[a_j] - h[a_{j+1}] + delta*(N/M)*N, 0)
        # = max(centroid[j] - centroid[j+1] + delta * N^2/M, 0)
        penalty_term = delta * (N / M) * N  # scalar = delta * N^2 / M

        # diff[j] = centroid[j] - centroid[j+1], j = 0..M-2
        diff = centroids[:-1] - centroids[1:]          # [M-1]
        violations = torch.clamp(diff + penalty_term, min=0.0)  # [M-1]

        # Normalize theo N (như paper: chia N ở mẫu)
        sample_loss = violations.sum() / N
        total_loss = total_loss + sample_loss
        valid_count += 1

    if valid_count == 0:
        return torch.zeros(1, device=device, dtype=dtype).squeeze()

    return (total_loss / valid_count).squeeze()
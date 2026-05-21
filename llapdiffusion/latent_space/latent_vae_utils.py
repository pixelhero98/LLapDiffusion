import torch


def compute_per_dim_stats(all_mu: torch.Tensor):
    """
    all_mu: [N, L, D]
    returns:
      mu_per_dim  [D], std_per_dim [D]  (std is clamped to >= 1e-6)
    """
    mu_per_dim  = all_mu.mean(dim=(0, 1))                         # [D]
    std_per_dim = all_mu.std(dim=(0, 1)).clamp(min=1e-6)          # [D]
    return mu_per_dim, std_per_dim


def normalize_and_check(all_mu: torch.Tensor, plot: bool = False, verbose: bool = True):
    """
    Per-dimension normalize and (optionally) plot a histogram.

    returns:
      all_mu_norm: [N, L, D]
      mu_per_dim:  [D]
      std_per_dim: [D]
    """
    mu_per_dim, std_per_dim = compute_per_dim_stats(all_mu)
    mu_b  = mu_per_dim.view(1, 1, -1)
    std_b = std_per_dim.view(1, 1, -1)
    all_mu_norm = (all_mu - mu_b) / std_b

    # global check
    all_vals = all_mu_norm.reshape(-1)
    if verbose:
        print(f"Global mean (post-norm): {all_vals.mean().item():.6f}")
        print(f"Global std  (post-norm): {all_vals.std().item():.6f}")

    # per-dim printout (first few)
    per_dim_mean = all_mu_norm.mean(dim=(0, 1))
    per_dim_std  = all_mu_norm.std(dim=(0, 1))
    D = all_mu_norm.size(-1)
    if verbose:
        print("\nPer-dim stats (first 10 dims or D if smaller):")
        for i in range(min(10, D)):
            print(f"  dim {i:2d}: mean={per_dim_mean[i]:7.4f}, std={per_dim_std[i]:7.4f}")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 3))
        plt.hist(all_vals.cpu().numpy(), bins=500, range=(-5, 5))
        plt.title("Histogram of normalized μ values")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.show()

    if verbose:
        print(f"NaNs: {torch.isnan(all_mu_norm).sum().item()} | Infs: {torch.isinf(all_mu_norm).sum().item()}")
        print(f"Min: {all_mu_norm.min().item():.6f} | Max: {all_mu_norm.max().item():.6f}")
    return all_mu_norm, mu_per_dim, std_per_dim

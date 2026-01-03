
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Guided Filter (He et al.) – used in paper to refine transmission
# ------------------------------------------------------------
def guided_filter(I, p, r=40, eps=1e-3):
    """
    I: guidance image (B,1,H,W) – grayscale RGB
    p: filtering input (B,1,H,W) – raw transmission
    """
    B, _, H, W = I.shape
    N = (2*r + 1) ** 2

    def box_filter(x):
        return F.avg_pool2d(x, kernel_size=2*r+1, stride=1, padding=r) * N

    mean_I = box_filter(I) / N
    mean_p = box_filter(p) / N
    mean_Ip = box_filter(I * p) / N
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box_filter(I * I) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a) / N
    mean_b = box_filter(b) / N

    q = mean_a * I + mean_b
    return q


# ------------------------------------------------------------
# Fog generator faithful to Sakaridis et al. (2017)
# ------------------------------------------------------------
class FogGenerator(nn.Module):
    """
    Implements:
    I(x) = R(x) t(x) + L (1 - t(x))
    t(x) = exp(-beta * depth)
    """
    def __init__(self, visibility_m=300.0):
        super().__init__()
        # beta = 2.996 / MOR (paper + meteorological definition)
        self.beta = 2.996 / visibility_m

    def forward(self, img, depth):
        """
        img   : (B,3,H,W) float in [0,1]
        depth : (B,1,H,W) depth in meters
        """
        B, _, H, W = img.shape
        device = img.device
        
        # ----------------------------------------------------
        # 1. Transmission from depth (homogeneous fog)
        # ----------------------------------------------------
        # Ensure depth > 0 (it should be since we filter, but just safe guard)
        t_raw = torch.exp(-self.beta * depth)
        t_raw = torch.clamp(t_raw, 0.0, 1.0)

        # ----------------------------------------------------
        # 2. Edge-preserving refinement (guided filtering)
        # ----------------------------------------------------
        gray = (0.299 * img[:,0:1] +
                0.587 * img[:,1:2] +
                0.114 * img[:,2:3])
        t = guided_filter(gray, t_raw)

        # ----------------------------------------------------
        # 3. Atmospheric light estimation (from farthest pixels)
        # ----------------------------------------------------
        depth_flat = depth.view(B, -1)
        img_flat = img.view(B, 3, -1)

        A = []
        k = int(0.001 * H * W)  # top 0.1% farthest pixels
        # Bound k for very small images
        k = max(10, k)
        
        for b in range(B):
            # Check indices validation: PyTorch topk
            if k >  depth_flat[b].numel():
                 k = depth_flat[b].numel()
                 
            idx = torch.topk(depth_flat[b], k, largest=True).indices
            A.append(img_flat[b,:,idx].mean(dim=1))
        
        A = torch.stack(A).view(B,3,1,1)

        # ----------------------------------------------------
        # 4. Fog synthesis (exact paper equation)
        # ----------------------------------------------------
        foggy = img * t + A * (1.0 - t)
        
        # Return only foggy image to match standard behavior, or tuple
        return foggy

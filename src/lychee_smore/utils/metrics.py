import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class Metrics:
    """
    A class to compute and track validation metrics including PSNR, LPIPS, LMD, SSIM using torchmetrics.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # 初始化 torchmetrics 指标
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
    
    def _normalize_for_display(self, tensor):
        """Convert from [-1, 1] to [0, 1] range."""
        return (tensor + 1.0) / 2.0
    
    def _convert_to_tensor(self, input_data):
        """
        Convert input to tensor format.
        
        Args:
            input_data: Can be:
                - torch.Tensor: shape (B, C, H, W) or (C, H, W) in range [-1, 1]
                - PIL.Image: single image
                - list[PIL.Image]: list of images
        
        Returns:
            torch.Tensor: shape (B, C, H, W) in range [-1, 1]
        """
        # Case 1: Already a tensor
        if isinstance(input_data, torch.Tensor):
            # If 3D tensor (C, H, W), add batch dimension
            if input_data.ndim == 3:
                input_data = input_data.unsqueeze(0)
            return input_data
        
        # Case 2: Single PIL Image
        elif isinstance(input_data, Image.Image):
            transform = transforms.Compose([
                transforms.ToTensor(),  # Converts to [0, 1]
                transforms.Normalize([0.5], [0.5])  # Converts to [-1, 1]
            ])
            tensor = transform(input_data).unsqueeze(0)  # Add batch dimension
            return tensor
        
        # Case 3: List of PIL Images
        elif isinstance(input_data, list) and all(isinstance(img, Image.Image) for img in input_data):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            tensors = [transform(img) for img in input_data]
            return torch.stack(tensors)
        
        else:
            raise TypeError(
                f"Unsupported input type: {type(input_data)}. "
                "Expected torch.Tensor, PIL.Image, or list[PIL.Image]"
            )
    
    def update(self, pred, target):
        """
        Compute and accumulate all metrics for a batch.
        
        Args:
            pred: Predicted images, can be:
                - torch.Tensor: shape (B, C, H, W) or (C, H, W) in range [-1, 1]
                - PIL.Image: single image
                - list[PIL.Image]: list of images
            target: Target images, same format options as pred
        """
        # Convert inputs to tensors
        pred_tensor = self._convert_to_tensor(pred)
        target_tensor = self._convert_to_tensor(target)
        
        # Ensure tensors are on the right device
        pred_tensor = pred_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        
        # Convert to [0, 1] range for PSNR and SSIM
        pred_norm = self._normalize_for_display(pred_tensor)
        target_norm = self._normalize_for_display(target_tensor)
        
        # Update torchmetrics
        self.psnr.update(pred_norm, target_norm)
        self.ssim.update(pred_norm, target_norm)
        self.lpips.update(pred_tensor, target_tensor)  # LPIPS expects [-1, 1] range
    
    def compute(self):
        """
        Compute final metrics across all accumulated data.
        Returns dictionary with metric values.
        """
        # Check if any samples have been processed
        if self.psnr.total == 0:
            return {
            'psnr': float('nan'),
            'ssim': float('nan'),
            'lpips': float('nan')
            }
        
        return {
            'psnr': self.psnr.compute().item(),
            'ssim': self.ssim.compute().item(), 
            'lpips': self.lpips.compute().item()
        }
    
    def reset(self):
        """Reset all metrics for new evaluation."""
        self.psnr.reset()
        self.ssim.reset()
        self.lpips.reset()
    
    def summarize(self):
        """
        Summarize all computed metrics.
        """
        metrics = self.compute()
        
        title = ""
        value = ""
        for metric_name, metric_value in metrics.items():
            title += f"{metric_name.upper():>10s}"
            value += f"{metric_value:>10.3f}"

        return title + "\n" + value

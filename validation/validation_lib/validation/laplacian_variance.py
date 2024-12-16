import torch
import torch.nn.functional as F
from torchmetrics import Metric


class LaplacianVarianceMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        # Track variance of the Laplacian for all images in the batch
        self.add_state("variances", default=[], dist_reduce_fx="cat")

    def update(self, image: torch.Tensor) -> None:
        """
        Update the metric with the current batch of images.

        Args:
            image (torch.Tensor): Batch of images, shape (B, C, H, W).
        """
        # Compute Laplacian variance for the image
        variance = self._laplacian_variance(image)
        self.variances.append(variance)

    def compute(self) -> torch.Tensor:
        """
        Compute the final sharpness metric (mean variance of Laplacian across all images).

        Returns:
            torch.Tensor: The mean Laplacian variance for the batch.
        """
        return torch.cat(self.variances).mean()

    def _laplacian_variance(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute the Laplacian variance for a batch of images.

        Args:
            image (torch.Tensor): A batch of images, shape (B, C, H, W).

        Returns:
            torch.Tensor: Variance of the Laplacian for each image in the batch.
        """
        # Ensure the image tensor is of float type
        image = image.float()  # Convert to float if it is not already

        # Define the Laplacian kernel (for single-channel)
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=image.device
        ).view(1, 1, 3, 3)

        # Move the kernel to the same device as the image
        laplacian_kernel = laplacian_kernel.to(image.device)

        # Apply the Laplacian kernel to each channel independently
        laplacian_list = []
        for c in range(image.size(1)):  # Loop over channels (C)
            channel_image = image[:, c : c + 1, :, :]  # Select the c-th channel
            laplacian = F.conv2d(channel_image, laplacian_kernel, padding=1)
            laplacian_list.append(laplacian)

        # Stack Laplacians from all channels and compute variance for each image
        laplacian_stack = torch.cat(laplacian_list, dim=1)  # Concatenate along channels dimension
        # Use reshape instead of view
        return laplacian_stack.reshape(laplacian_stack.size(0), -1).var(dim=1)  # Variance across pixels

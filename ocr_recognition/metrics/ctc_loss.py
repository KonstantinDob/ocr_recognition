import torch
import torch.nn.functional as F
from torch.nn import CTCLoss


class CTCLossOCR(CTCLoss):
    """Create CTC Loss to be suitable for Gyomei trainer."""

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Prepare Gyomei input to torch CTC Loss.

        Args:
            log_probs (torch.tensor): NN output.
            targets (torch.tensor): Target. Consist of target and CTC length data.

        Returns:
            torch.nn.Module: CTC Loss value.
        """
        device = log_probs.device.type
        input_lengths = torch.full(
            size=(log_probs.size(0),), fill_value=log_probs.size(1), dtype=torch.long
        ).to(device)
        target_lengths = torch.squeeze(targets[:, -1:]).to(torch.long)
        targets = targets[:, :-1].to(torch.long)

        return F.ctc_loss(
            log_probs.log_softmax(2).permute(1, 0, 2),
            targets,
            input_lengths,
            target_lengths,
            self.blank,
            self.reduction,
            self.zero_infinity
        )

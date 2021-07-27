# third party
import torch


def sqrt_PEHE(y: torch.Tensor, hat_y: torch.Tensor) -> torch.Tensor:
    """
    Precision in Estimation of Heterogeneous Effect(PyTorch version).
    PEHE reflects the ability to capture individual variation in treatment effects.
    Args:
        y: expected outcome.
        hat_y: estimated outcome.
    """
    y = torch.Tensor(y)
    hat_y = torch.Tensor(hat_y)
    return torch.sqrt(torch.mean(((y[:, 1] - y[:, 0]) - hat_y) ** 2))


def ATE(y: torch.Tensor, hat_y: torch.Tensor) -> torch.Tensor:
    """
    Average Treatment Effect.
    ATE measures what is the expected causal effect of the treatment across all individuals in the population.
    Args:
        y: expected outcome.
        hat_y: estimated outcome.
    """
    y = torch.Tensor(y)
    hat_y = torch.Tensor(hat_y)
    return torch.abs(
        torch.mean(y[:, 1] - y[:, 0]) - torch.mean(hat_y[:, 1] - hat_y[:, 0])
    )

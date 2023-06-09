import torch
import torch.nn as nn

class R1(nn.Module):
    """
    Implementation of the R1 GAN regularization.
    """

    def __init__(self, args):
        """
        Constructor method
        """
        # Call super constructor
        super(R1, self).__init__()
        self.args = args

    def forward(self, prediction_real: torch.Tensor, real_sample: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the regularization
        :param prediction_real: (torch.Tensor) Prediction of the discriminator for a batch of real images
        :param real_sample: (torch.Tensor) Batch of the corresponding real images
        :return: (torch.Tensor) Loss value
        """
        # Calc gradient
        grad_real = torch.autograd.grad(outputs=prediction_real.sum(), inputs=real_sample, create_graph=True)[0]
        # Calc regularization
        regularization_loss: torch.Tensor = self.args.lambda_r1 \
                                            * grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return regularization_loss
import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """

    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError(
                "Dimension to apply softmax to is greater than the number of dimensions in Z"
            )

        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        exp_Z = np.exp(Z - np.max(Z, axis=self.dim, keepdims=True))
        self.A = exp_Z / (np.sum(exp_Z, axis=self.dim, keepdims=True) + 1e-9)

        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass

        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]

        dLdA_permuted = np.moveaxis(dLdA, self.dim, -1)
        dLdA_flat = dLdA_permuted.reshape(-1, C)
        A_permuted = np.moveaxis(self.A, self.dim, -1)
        A_flat = A_permuted.reshape(-1, C)

        dLdZ_flat = np.zeros_like(A_flat)

        for i in range(dLdA_flat.shape[0]):
            # Initialize the Jacobian with all zeros.
            # Hint: Jacobian matrix for softmax is a _×_ matrix, but what is _ here?
            J = np.zeros((C, C))

            # Fill the Jacobian matrix, please read the writeup for the conditions.
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m, n] = A_flat[i, m] * (1 - A_flat[i, m])
                    else:
                        J[m, n] = -1.0 * A_flat[i, m] * A_flat[i, n]

            # Calculate the derivative of the loss with respect to the i-th input, please read the writeup for it.
            # Hint: How can we use (1×C) and (C×C) to get (1×C) and stack up vertically to give (N×C) derivative matrix?
            dLdZ_flat[i, :] = dLdA_flat[i, :] @ J

        # Reshape input to 2D
        # if len(shape) > 2:
        # self.A = NotImplementedError
        # dLdA = NotImplementedError

        # Reshape back to original dimensions if necessary
        # if len(shape) > 2:
        #     # Restore shapes to original
        #     self.A = NotImplementedError
        #     dLdZ = NotImplementedError

        dLdZ = dLdZ_flat.reshape(dLdA_permuted.shape)
        dLdZ = np.moveaxis(dLdZ, -1, self.dim)

        return dLdZ

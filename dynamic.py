import numpy as np
import hashlib
from tpm import TPM

class DynamicTPM(TPM):
    def remap_weights(self, old_l, new_l):
        # Deterministic Proportional Rescale
        self.weights = np.round(self.weights * (new_l / old_l)).astype(int)
        self.l = new_l

    def optimize(self, x, tau_remote, use_modulo=False):
        tau, sigma = self.get_output(x)
        if tau == tau_remote:
            x_reshaped = x.reshape(self.k, self.n)
            updates = x_reshaped * tau * (sigma.reshape(-1, 1) == tau)
            
            if use_modulo:
                # Modulo Rule for Flatness
                range_size = 2 * self.l + 1
                self.weights = ((self.weights + updates + self.l) % range_size) - self.l
            else:
                # Standard Clipping for Speed
                self.weights = np.clip(self.weights + updates, -self.l, self.l)

    def get_hashed_key(self):
        # Final high-entropy extraction
        return hashlib.sha256(self.weights.tobytes()).hexdigest()
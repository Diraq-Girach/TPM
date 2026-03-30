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

    def get_von_neumann_key(self):
        
        flat = self.weights.flatten()

        # Step 1: sign bits, dropping zeros
        sign_bits = [1 if w > 0 else 0 for w in flat if w != 0]

        # Step 2: Von Neumann extraction over consecutive pairs
        extracted = []
        i = 0
        while i + 1 < len(sign_bits):
            a, b = sign_bits[i], sign_bits[i + 1]
            if a != b:                  # (0,1) → 1 ; (1,0) → 0
                extracted.append(b)
            i += 2                      # always advance by 2 (pairs are consumed)

        raw_bits = ''.join(str(b) for b in extracted)

        if not extracted:
            # Degenerate case: all weights identical — return a zero digest
            return hashlib.sha256(b'\x00').hexdigest(), raw_bits

        # Step 3: pack bits → bytes → SHA-256
        # Pad to nearest multiple of 8
        padded = raw_bits.ljust((len(raw_bits) + 7) // 8 * 8, '0')
        byte_array = bytes(
            int(padded[j:j + 8], 2) for j in range(0, len(padded), 8)
        )
        hashed = hashlib.sha256(byte_array).hexdigest()
        return hashed, raw_bits
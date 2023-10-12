import math
import random
import numpy as np

class MaskingGenerator:
    def __init__(
            self, min_num_patches=4, max_patch_prob=0.4,
            min_aspect=0.3, max_aspect=None):

        self.min_num_patches = min_num_patches
        self.max_proportion_patches = max_patch_prob

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _mask(self, mask, max_mask_patches, height, width):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, self.max_proportion_patches * (height*width))
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < width and h < height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, height, width):
        mask = np.zeros(shape=(height, width), dtype=np.int)
        mask_count = 0
        num_masked_patches = self.max_proportion_patches * (height*width)
        while mask_count < num_masked_patches:
            max_mask_patches = num_masked_patches - mask_count
            max_mask_patches = min(max_mask_patches, num_masked_patches)

            delta = self._mask(mask, max_mask_patches, height, width)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask
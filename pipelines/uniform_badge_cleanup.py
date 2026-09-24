"""Conservative removal of template-matched vivid badges on dark fabric."""
import cv2
import numpy as np
from PIL import Image


def _patches(rgb, hue=None, allow_bottom_edge=False):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = rgb.shape[:2]
    selected = (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 125)
    selected[:int(h * .45)] = False
    if hue is not None:
        delta = np.abs(hsv[:, :, 0].astype(float) - hue)
        selected &= np.minimum(delta, 180 - delta) < 10
    count, labels, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8), 8)
    patches = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if not (.0008 < area / (h * w) < .025 and .65 < bw / max(bh, 1) < 1.5):
            continue
        if x < 5 or y < 5 or x + bw >= w - 5 or (not allow_bottom_edge and y + bh >= h - 5):
            continue
        mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)
        if mask.sum() / (bw * bh) < .5:
            continue
        pad = max(3, round(min(bw, bh) * .12))
        expanded = cv2.dilate(mask, np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8))
        outer = cv2.dilate(expanded, np.ones((9, 9), np.uint8))
        ring = (outer > 0) & (expanded == 0)
        if not ring.any() or np.percentile(hsv[:, :, 2][ring], 75) > 110:
            continue
        patches.append((expanded, float(np.median(hsv[:, :, 0][labels == i]))))
    return patches


def remove_template_badges(image: Image.Image, template: Image.Image):
    """Return corrected RGB and count; leave unsupported badge types untouched.

    Color alone is insufficient: each matched patch must be compact, interior
    to the lower image, and surrounded by dark cloth. No face/hair edits.
    """
    reference = np.array(template.convert("RGB"))
    candidates = _patches(reference, allow_bottom_edge=True)
    rgb = np.array(image.convert("RGB"))
    mask = np.zeros(rgb.shape[:2], np.uint8)
    count = 0
    for _, hue in candidates:
        for patch, _ in _patches(rgb, hue, allow_bottom_edge=True):
            if not np.any(mask[patch > 0]):
                count += 1
            mask[patch > 0] = 255
    if not count:
        return image.convert("RGB"), 0
    cleaned = cv2.inpaint(rgb, mask, 5, cv2.INPAINT_TELEA)
    cleaned[mask == 0] = rgb[mask == 0]
    return Image.fromarray(cleaned), count

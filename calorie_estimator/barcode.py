"""Barcode detection from image bytes.

Wraps ``pyzbar`` to decode UPC/EAN barcodes. The detector is called by
``CalorieEstimator`` before the vision pipeline: if a barcode is present
we can short-circuit to an Open Food Facts lookup and skip the LLM
entirely.

Install:
    pip install pyzbar pillow
    # Plus the zbar system library:
    #   macOS: brew install zbar
    #   Debian/Ubuntu: apt-get install libzbar0

The module imports lazily so the rest of the package still works in
environments where pyzbar or zbar aren't installed — ``detect_barcodes``
simply returns an empty list and logs a warning in that case.
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

# Barcode symbologies we trust to identify a packaged-food product.
# Other types (QR, Code-39, Code-128, etc.) are ignored — they don't
# correspond to product UPCs and would produce garbage OFF lookups.
_FOOD_SYMBOLOGIES = {"EAN13", "EAN8", "UPCA", "UPCE"}

_IMPORT_FAILURE_LOGGED = False


def detect_barcodes(image: bytes) -> list[str]:
    """Decode any UPC/EAN barcodes visible in ``image``.

    Returns the distinct list of decoded data strings, in the order
    pyzbar found them. Returns an empty list when:
    - pyzbar or Pillow isn't installed
    - the image can't be decoded
    - no supported barcode is detected
    """
    global _IMPORT_FAILURE_LOGGED

    try:
        from PIL import Image
        from pyzbar import pyzbar
    except ImportError as e:
        if not _IMPORT_FAILURE_LOGGED:
            logger.warning(
                "Barcode detection unavailable (%s). "
                "Install with: pip install pyzbar pillow "
                "(and the zbar system library).",
                e,
            )
            _IMPORT_FAILURE_LOGGED = True
        return []

    try:
        with Image.open(io.BytesIO(image)) as pil_image:
            pil_image.load()
            results = pyzbar.decode(pil_image)
    except Exception as e:
        logger.warning(f"Barcode decode failed: {e}")
        return []

    seen: set[str] = set()
    ordered: list[str] = []
    for r in results:
        symbology = getattr(r.type, "name", r.type)
        if symbology not in _FOOD_SYMBOLOGIES:
            continue
        try:
            code = r.data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if code and code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered

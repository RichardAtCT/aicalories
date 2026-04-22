"""Open Food Facts API client.

Looks up packaged foods by UPC/EAN barcode and optionally submits new
product nutrition back to OFF after a user re-shoots a missing product's
label.

OFF is community-maintained and has especially strong coverage in Europe;
coverage in other regions is patchier. The library treats a missing
product or missing ``nutriments`` block as a failure that the caller
should surface to the user (prompt for a label photo).

API docs: https://openfoodfacts.github.io/openfoodfacts-server/api/
Write API: https://openfoodfacts.github.io/openfoodfacts-server/api/tutorial-adding-a-product/
"""

from __future__ import annotations

import logging
import os

import httpx

from .models import NutrientProfile, OFFContribution, OpenFoodFactsProduct

logger = logging.getLogger(__name__)

OFF_BASE_URL_PROD = "https://world.openfoodfacts.org"
OFF_BASE_URL_STAGING = "https://world.openfoodfacts.net"
DEFAULT_USER_AGENT = "aicalories/0.1 (+https://github.com/RichardAtCT/aicalories)"


class OpenFoodFactsClient:
    """Async client for the Open Food Facts product API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 10.0,
        user_agent: str = DEFAULT_USER_AGENT,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("OFF_BASE_URL")
            or OFF_BASE_URL_PROD
        ).rstrip("/")
        self.timeout = timeout
        self.user_agent = user_agent
        self.username = username or os.environ.get("OFF_USERNAME", "")
        self.password = password or os.environ.get("OFF_PASSWORD", "")

    # ── Read ─────────────────────────────────────────────────

    async def lookup(self, barcode: str) -> OpenFoodFactsProduct | None:
        """Fetch product data by barcode.

        Returns ``None`` when the product isn't in OFF (the caller should
        then prompt the user to re-shoot the nutrition label).
        """
        if not barcode:
            return None

        url = f"{self.base_url}/api/v2/product/{barcode}.json"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            ) as client:
                response = await client.get(url)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            logger.warning(f"OFF lookup failed for {barcode}: {e}")
            return None

        # OFF returns {"status": 1, "product": {...}} on hit, {"status": 0} on miss.
        if data.get("status") != 1:
            return None

        product = data.get("product") or {}
        return _parse_product(barcode, product)

    # ── Write ────────────────────────────────────────────────

    async def submit(self, contribution: OFFContribution) -> bool:
        """Submit a new/updated product record to Open Food Facts.

        Uses ``product_jqm2.pl``, the long-standing write endpoint that
        accepts form-encoded fields. Anonymous edits are allowed (OFF
        logs them against the IP); set ``OFF_USERNAME`` / ``OFF_PASSWORD``
        to attribute to a real account.

        Returns ``True`` on OFF ``status == 1``, else ``False``. The caller
        is responsible for getting explicit user confirmation before
        calling this method — the library never submits on its own.
        """
        if not contribution.is_submittable():
            logger.warning(
                "Refusing to submit partial OFF contribution for %s",
                contribution.barcode,
            )
            return False

        n = contribution.nutrients_per_100g
        form: dict[str, str] = {
            "code": contribution.barcode,
            "product_name": contribution.product_name,
            "brands": contribution.brand,
            "nutrition_data_per": "100g",
            "nutriment_energy-kcal_100g": _fmt(n.calories),
            "nutriment_energy-kcal_unit": "kcal",
            "nutriment_proteins_100g": _fmt(n.protein_g),
            "nutriment_fat_100g": _fmt(n.fat_g),
            "nutriment_carbohydrates_100g": _fmt(n.carbs_g),
            "nutriment_fiber_100g": _fmt(n.fiber_g),
            "nutriment_sugars_100g": _fmt(n.sugar_g),
            "nutriment_salt_100g": _fmt(n.sodium_mg / 1000.0 * 2.5),
            "nutriment_saturated-fat_100g": _fmt(n.saturated_fat_g),
        }
        if contribution.serving_size_label:
            form["serving_size"] = contribution.serving_size_label
        if contribution.serving_quantity_g is not None:
            form["serving_quantity"] = _fmt(contribution.serving_quantity_g)
        if self.username and self.password:
            form["user_id"] = self.username
            form["password"] = self.password
        form["comment"] = "Added via aicalories label scan"

        url = f"{self.base_url}/cgi/product_jqm2.pl"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            ) as client:
                response = await client.post(url, data=form)
                response.raise_for_status()
                body = response.json()
        except Exception as e:
            logger.warning(
                "OFF submit failed for %s: %s", contribution.barcode, e
            )
            return False

        ok = body.get("status") == 1 or body.get("status_verbose") == "fields saved"
        if not ok:
            logger.warning("OFF rejected submission: %s", body)
        return ok


# ── Parsing helpers ──────────────────────────────────────────


def _parse_product(barcode: str, product: dict) -> OpenFoodFactsProduct | None:
    """Turn an OFF /api/v2/product payload into an OpenFoodFactsProduct."""
    try:
        nutriments = product.get("nutriments") or {}
        nutrients = NutrientProfile(
            calories=_nutri_kcal(nutriments),
            protein_g=_nf(nutriments.get("proteins_100g")),
            fat_g=_nf(nutriments.get("fat_100g")),
            carbs_g=_nf(nutriments.get("carbohydrates_100g")),
            fiber_g=_nf(nutriments.get("fiber_100g")),
            sugar_g=_nf(nutriments.get("sugars_100g")),
            # OFF stores sodium in g/100g; our model stores mg/100g.
            sodium_mg=_nf(nutriments.get("sodium_100g")) * 1000.0,
            saturated_fat_g=_nf(nutriments.get("saturated-fat_100g")),
        )
        serving_quantity = product.get("serving_quantity")
        try:
            serving_quantity_g = float(serving_quantity) if serving_quantity else None
        except (TypeError, ValueError):
            serving_quantity_g = None

        return OpenFoodFactsProduct(
            barcode=barcode,
            product_name=(product.get("product_name") or "").strip() or "Unknown product",
            brand=(product.get("brands") or "").split(",")[0].strip(),
            serving_size_label=(product.get("serving_size") or "").strip(),
            serving_quantity_g=serving_quantity_g,
            nutrients_per_100g=nutrients,
            data_quality_warnings=list(product.get("data_quality_warnings_tags") or []),
        )
    except Exception as e:
        logger.debug(f"Failed to parse OFF product {barcode}: {e}")
        return None


def _nutri_kcal(nutriments: dict) -> float:
    """Extract energy in kcal from an OFF nutriments block.

    OFF may report energy in either kJ ("energy_100g") or kcal
    ("energy-kcal_100g"). Prefer kcal when present.
    """
    kcal = nutriments.get("energy-kcal_100g")
    if kcal is not None:
        return _nf(kcal)
    kj = nutriments.get("energy_100g")
    if kj is not None:
        return round(_nf(kj) / 4.184, 1)
    return 0.0


def _nf(value) -> float:
    """Coerce an OFF field to a float, treating ``None``/strings as 0."""
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: float) -> str:
    """Format a float for OFF form-encoding (short, no scientific notation)."""
    return f"{round(float(value), 3):g}"

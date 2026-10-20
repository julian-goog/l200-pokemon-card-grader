"""PII and sensitive credential redaction module for PokéGrade AI.

Implements active scrubbing mechanisms to redact sensitive collector data
(shipping addresses, PayPal/personal emails, credit card numbers, phone numbers,
IP addresses, and API tokens) before logging or persistent storage using:
1. Google Cloud Sensitive Data Protection (Cloud DLP API — `google.cloud.dlp_v2`)
   when live GCP DLP inspection/de-identification is enabled.
2. Deterministic compiled regex scrubbing ensuring zero-latency protection in
   both cloud and local/offline execution environments.
"""

from __future__ import annotations

import os
import re
from typing import Any

# Compiled regular expressions ordered from most specific to general
_API_TOKEN_PATTERN = re.compile(
    r"(?:"
    r"AIza[0-9A-Za-z\-_]{16,64}"
    r"|sk-[0-9A-Za-z\-_]{16,64}"
    r"|tcg_(?:live|test)_[0-9A-Za-z\-_]{8,64}"
    r"|ya29\.[0-9A-Za-z\-_]+"
    r"|Bearer\s+[0-9A-Za-z\-_\.]{12,}"
    r"|(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[0-9A-Za-z\-_]{8,}['\"]?"
    r")",
    re.IGNORECASE,
)

_CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{15,16}\b"
)

_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

_SHIPPING_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\-]+(?:\s+[A-Za-z0-9.\-]+){0,4}\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Ter|Circle|Cir|Parkway|Pkwy)\b"
    r"(?:(?:,\s*|\s+)(?:Apt|Suite|Ste|Unit|#)\s*[A-Za-z0-9\-]+)?"
    r"(?:,\s*[A-Za-z\s]+(?:,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?)?",
    re.IGNORECASE,
)

_IPV4_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)

_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)

_SENSITIVE_KEY_NAMES = frozenset(
    {
        "shipping_address",
        "buyer_address",
        "seller_address",
        "paypal_email",
        "seller_email",
        "buyer_email",
        "credit_card",
        "card_number",
        "cvv",
        "api_key",
        "secret_key",
        "access_token",
        "authorization",
    }
)

DEFAULT_DLP_INFO_TYPES: tuple[dict[str, str], ...] = (
    {"name": "EMAIL_ADDRESS"},
    {"name": "CREDIT_CARD_NUMBER"},
    {"name": "STREET_ADDRESS"},
    {"name": "PHONE_NUMBER"},
    {"name": "IP_ADDRESS"},
    {"name": "AUTH_TOKEN"},
    {"name": "GCP_API_KEY"},
)


class CloudDLPRedactor:
    """Google Cloud Sensitive Data Protection (DLP API) wrapper with regex fallback."""

    def __init__(
        self,
        project_id: str | None = None,
        enable_cloud_dlp_api: bool | None = None,
    ) -> None:
        """Initialize the Cloud DLP redactor.

        Args:
            project_id: Google Cloud project ID for `projects/{project_id}/locations/global`.
            enable_cloud_dlp_api: Whether to invoke live `google.cloud.dlp_v2` API calls.
                Defaults to checking `ENABLE_CLOUD_DLP_API=true` in environment.
        """
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "pokegrade-gcp-prod")
        if enable_cloud_dlp_api is None:
            self.enable_cloud_dlp_api = (
                os.environ.get("ENABLE_CLOUD_DLP_API", "false").lower() == "true"
            )
        else:
            self.enable_cloud_dlp_api = enable_cloud_dlp_api

    def deidentify_with_cloud_dlp(self, text: str) -> str:
        """De-identify sensitive PII using Google Cloud DLP API (`google.cloud.dlp_v2`).

        Falls back gracefully to local deterministic regex redaction if `google-cloud-dlp`
        is not installed or when running in offline test mode.

        Args:
            text: Input string potentially containing PII.

        Returns:
            De-identified string from Cloud DLP API followed by local regex scrubbing.
        """
        if not text or not self.enable_cloud_dlp_api:
            return _regex_scrub(text)

        try:
            from google.cloud import dlp_v2  # type: ignore[import-untyped]

            client = dlp_v2.DlpServiceClient()
            parent = f"projects/{self.project_id}/locations/global"
            inspect_config = {
                "info_types": list(DEFAULT_DLP_INFO_TYPES),
                "min_likelihood": "POSSIBLE",
            }
            deidentify_config = {
                "info_type_transformations": {
                    "transformations": [
                        {
                            "primitive_transformation": {
                                "replace_with_info_type_config": {}
                            }
                        }
                    ]
                }
            }
            item = {"value": text}
            response = client.deidentify_content(
                request={
                    "parent": parent,
                    "inspect_config": inspect_config,
                    "deidentify_config": deidentify_config,
                    "item": item,
                }
            )
            return _regex_scrub(str(response.item.value))
        except Exception:
            return _regex_scrub(text)


def _regex_scrub(text: str) -> str:
    """Apply deterministic regex masks for tokens, cards, emails, addresses, IPs, and phones."""
    if not isinstance(text, str) or not text:
        return text

    scrubbed = _API_TOKEN_PATTERN.sub("[REDACTED_API_TOKEN]", text)
    scrubbed = _CREDIT_CARD_PATTERN.sub("[REDACTED_CREDIT_CARD]", scrubbed)
    scrubbed = _EMAIL_PATTERN.sub("[REDACTED_EMAIL_PII]", scrubbed)
    scrubbed = _SHIPPING_ADDRESS_PATTERN.sub("[REDACTED_SHIPPING_ADDRESS]", scrubbed)
    scrubbed = _IPV4_PATTERN.sub("[REDACTED_IP]", scrubbed)
    scrubbed = _PHONE_PATTERN.sub("[REDACTED_PHONE]", scrubbed)
    return scrubbed


_DEFAULT_DLP_REDACTOR = CloudDLPRedactor()


def redact_pii(text: str) -> str:
    """Scrub PII, payment details, shipping addresses, and API tokens from text.

    Routes through `CloudDLPRedactor` (Google Cloud DLP API + compiled regex patterns)
    to guarantee sensitive data is sanitized before logging or persistent storage.

    Args:
        text: Raw input string that may contain collector PII or secrets.

    Returns:
        Sanitized string with sensitive substrings replaced by explicit redaction tokens.
    """
    if not isinstance(text, str) or not text:
        return text
    return _DEFAULT_DLP_REDACTOR.deidentify_with_cloud_dlp(text)


def redact_payload(payload: Any) -> Any:
    """Recursively sanitize dictionaries, lists, and primitives for safe logging/storage.

    Args:
        payload: Arbitrary Python data structure (str, dict, list, tuple, or primitive).

    Returns:
        Deep-copied data structure with all sensitive strings and keys redacted.
    """
    if isinstance(payload, str):
        return redact_pii(payload)
    if isinstance(payload, dict):
        sanitized: dict[Any, Any] = {}
        for key, value in payload.items():
            key_lower = str(key).lower()
            if key_lower in _SENSITIVE_KEY_NAMES and isinstance(value, str):
                if "address" in key_lower:
                    sanitized[key] = "[REDACTED_SHIPPING_ADDRESS]"
                elif "email" in key_lower:
                    sanitized[key] = "[REDACTED_EMAIL_PII]"
                elif key_lower == "card_number":
                    digits_only = "".join(ch for ch in value if ch.isdigit())
                    if len(digits_only) >= 12 or _CREDIT_CARD_PATTERN.search(value):
                        sanitized[key] = "[REDACTED_CREDIT_CARD]"
                    else:
                        sanitized[key] = redact_pii(value)
                elif "card" in key_lower or "cvv" in key_lower:
                    sanitized[key] = "[REDACTED_CREDIT_CARD]"
                else:
                    sanitized[key] = "[REDACTED_API_TOKEN]"
            else:
                sanitized[key] = redact_payload(value)
        return sanitized
    if isinstance(payload, list):
        return [redact_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(redact_payload(item) for item in payload)
    return payload

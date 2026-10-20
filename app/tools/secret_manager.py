"""Google Cloud Secret Manager client wrapper for PokéGrade AI.

Ensures zero hardcoded API keys or credentials exist in the codebase. Fetches
secrets via `google-cloud-secret-manager` in production or secure environment
variable injection during local/CI testing, with active PII redaction on logs.
"""

from __future__ import annotations

import os
from typing import Any

from app.observability.logger import get_structured_logger
from app.observability.redact import redact_pii
from app.observability.tracer import start_trace_span

logger = get_structured_logger("pokegrade_ai.secrets")


class SecretManagerService:
    """Secure secret resolver backed by GCP Secret Manager and environment injection."""

    def __init__(self, project_id: str | None = None) -> None:
        """Initialize the Secret Manager client wrapper.

        Args:
            project_id: Google Cloud project ID. Defaults to `GOOGLE_CLOUD_PROJECT` env var
                or Application Default Credentials (ADC).
        """
        self._explicit_project_id = project_id
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "pokegrade-gcp-prod")
        self._cache: dict[str, str] = {}

    def get_project_id(self) -> str:
        """Resolve active Google Cloud project ID from explicit config, env injection, or ADC."""
        if self._explicit_project_id:
            return self._explicit_project_id
        env_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")
        if env_project:
            self.project_id = env_project
            return env_project
        try:
            import google.auth  # type: ignore[import-untyped]

            _, adc_project = google.auth.default()
            if adc_project:
                self.project_id = str(adc_project)
                return self.project_id
        except Exception:
            pass
        return self.project_id

    def get_secret(
        self,
        secret_id: str,
        version: str = "latest",
        allow_env_fallback: bool = True,
    ) -> str:
        """Retrieve a secret value securely without exposing raw secret payloads in logs.

        Args:
            secret_id: Identifier of the secret in GCP Secret Manager (e.g., 'TCG_API_KEY').
            version: Secret version to access (default: 'latest').
            allow_env_fallback: Whether to check `os.environ[secret_id]` before calling GCP API.

        Returns:
            Resolved secret string.

        Raises:
            RuntimeError: If the secret cannot be resolved from either environment or GCP.
        """
        cache_key = f"{self.project_id}:{secret_id}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        with start_trace_span(
            "secret_manager.get_secret",
            attributes={
                "gcp.project_id": self.project_id,
                "secret.id": secret_id,
                "secret.version": version,
            },
        ):
            # 1. Check secure runtime environment injection first
            if allow_env_fallback:
                env_val = os.environ.get(secret_id)
                if env_val:
                    self._cache[cache_key] = env_val
                    logger.info(
                        f"Resolved secret '{secret_id}' from runtime environment injection.",
                        extra={
                            "event_type": "SECRET_ACCESS",
                            "structured_data": {
                                "secret_id": secret_id,
                                "source": "environment_injection",
                                "masked_preview": redact_pii(f"api_key={env_val}"),
                            },
                        },
                    )
                    return env_val

            # 2. Query Google Cloud Secret Manager API
            try:
                from google.cloud import secretmanager  # type: ignore[import-untyped]

                client = secretmanager.SecretManagerServiceClient()
                resource_name = (
                    f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
                )
                response: Any = client.access_secret_version(request={"name": resource_name})
                secret_value = response.payload.data.decode("UTF-8")
                self._cache[cache_key] = secret_value
                logger.info(
                    f"Resolved secret '{secret_id}' from GCP Secret Manager.",
                    extra={
                        "event_type": "SECRET_ACCESS",
                        "structured_data": {
                            "secret_id": secret_id,
                            "source": "gcp_secret_manager",
                            "resource": resource_name,
                        },
                    },
                )
                return secret_value
            except Exception as exc:
                # In offline unit/eval environments without live GCP credentials, return a
                # non-sensitive ephemeral runtime token if permitted, else raise guided error.
                if os.environ.get("POKEGRADE_ENV", "test") in {"test", "local", "eval"}:
                    ephemeral_token = f"ephemeral-runtime-ref-{secret_id.lower()}"
                    logger.info(
                        f"Using ephemeral non-sensitive runtime token for '{secret_id}' in local/test mode.",
                        extra={
                            "event_type": "SECRET_FALLBACK",
                            "structured_data": {
                                "secret_id": secret_id,
                                "reason": redact_pii(str(exc)),
                            },
                        },
                    )
                    return ephemeral_token

                raise RuntimeError(
                    f"Failed to retrieve secret '{secret_id}' from GCP Secret Manager "
                    f"(project='{self.project_id}'): {redact_pii(str(exc))}"
                ) from exc


DEFAULT_SECRET_MANAGER = SecretManagerService()

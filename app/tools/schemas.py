"""Strict Pydantic v2 schemas and domain Enums for PokéGrade AI tools.

Enforces explicit JSON schema contracts, numeric subgrade boundaries (1.0–10.0),
domain enumerations (`Rarity`, `GradingCompany`, `Condition`), and structured
error recovery models across all agent tool invocations.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Rarity(str, Enum):
    """Official Pokémon TCG card rarity tiers."""

    COMMON = "COMMON"
    UNCOMMON = "UNCOMMON"
    HOLO_RARE = "HOLO_RARE"
    ULTRA_RARE = "ULTRA_RARE"
    ILLUSTRATION_RARE = "ILLUSTRATION_RARE"
    SPECIAL_ILLUSTRATION_RARE = "SPECIAL_ILLUSTRATION_RARE"
    SECRET_RARE = "SECRET_RARE"


class GradingCompany(str, Enum):
    """Supported third-party collectible card grading authorities."""

    PSA = "PSA"
    BGS = "BGS"
    CGC = "CGC"


class Condition(str, Enum):
    """Standardized collectible card condition classifications."""

    GEM_MINT = "GEM_MINT"
    MINT = "MINT"
    NEAR_MINT_MINT = "NEAR_MINT_MINT"
    NEAR_MINT = "NEAR_MINT"
    EXCELLENT_MINT = "EXCELLENT_MINT"
    EXCELLENT = "EXCELLENT"
    LIGHTLY_PLAYED = "LIGHTLY_PLAYED"
    MODERATELY_PLAYED = "MODERATELY_PLAYED"
    HEAVILY_PLAYED = "HEAVILY_PLAYED"
    DAMAGED = "DAMAGED"


class CardSubgradesInput(BaseModel):
    """Input schema for physical Pokémon card subgrade evaluation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    card_name: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Official Pokémon card name (e.g., 'Charizard Holo', 'Umbreon VMAX Alt Art').",
    )
    set_name: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Expansion set name (e.g., 'Base Set 1999', 'Evolving Skies').",
    )
    card_number: str = Field(
        default="001/102",
        min_length=1,
        max_length=32,
        description="Collector set number printed on the card (e.g., '4/102', '215/203').",
    )
    rarity: Rarity = Field(
        default=Rarity.HOLO_RARE,
        description="Official rarity tier of the card.",
    )
    centering_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Front/back border centering score on a 1.0 to 10.0 scale.",
    )
    corners_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Sharpness and whitening score across all four corners (1.0 to 10.0).",
    )
    edges_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Edge chipping, silvering, and smoothness score (1.0 to 10.0).",
    )
    surface_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Holo scratching, print lines, indents, or crease score (1.0 to 10.0).",
    )
    grading_company: GradingCompany = Field(
        default=GradingCompany.PSA,
        description="Target grading service authority (PSA, BGS, or CGC).",
    )
    surface_notes: str = Field(
        default="",
        max_length=500,
        description="Optional physical defect notes (e.g., 'micro holo scratch', 'heavy crease').",
    )


class CardSubgradesOutput(BaseModel):
    """Output schema produced by `analyze_card_condition_subgrades`."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        default="SUCCESS",
        description="Execution status indicator ('SUCCESS').",
    )
    card_identifier: str = Field(
        ...,
        description="Normalized identifier combining card name, set, and collector number.",
    )
    estimated_grade: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Conservative composite numerical grade capped by the lowest subgrade.",
    )
    condition_tier: Condition = Field(
        ...,
        description="Standardized condition classification corresponding to the composite grade.",
    )
    limiting_subgrade_factor: str = Field(
        ...,
        description="The weakest physical dimension constraining the final grade.",
    )
    subgrade_breakdown: dict[str, float] = Field(
        ...,
        description="Dictionary mapping centering, corners, edges, and surface to their scores.",
    )
    weighted_raw_average: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Unconstrained weighted average prior to conservative subgrade cap.",
    )
    grading_company: GradingCompany = Field(
        ...,
        description="Grading authority rules applied during calculation.",
    )
    submission_eligible: bool = Field(
        ...,
        description="True if the card condition meets minimum threshold (>= 7.0) for grading consideration.",
    )
    inspector_notes: str = Field(
        ...,
        description="Detailed technical explanation of how the grade was derived.",
    )


class MarketPricingQueryInput(BaseModel):
    """Input schema for TCG market valuation and grading ROI analysis."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    card_name: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Official Pokémon card name to look up in the pricing index.",
    )
    set_name: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Expansion set name for accurate market matching.",
    )
    rarity: Rarity = Field(
        default=Rarity.HOLO_RARE,
        description="Card rarity tier used for baseline market valuation.",
    )
    estimated_grade: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Projected numerical grade (1.0–10.0) from the Card Inspector Subagent.",
    )
    grading_company: GradingCompany = Field(
        default=GradingCompany.PSA,
        description="Grading service whose fee schedule and market multiplier apply.",
    )


class MarketPricingOutput(BaseModel):
    """Output schema produced by `query_tcg_market_pricing`."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        default="SUCCESS",
        description="Execution status indicator ('SUCCESS').",
    )
    card_name: str = Field(..., description="Evaluated Pokémon card name.")
    set_name: str = Field(..., description="Evaluated Pokémon expansion set.")
    grading_company: GradingCompany = Field(..., description="Selected grading company.")
    estimated_grade: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Projected grade evaluated for market pricing.",
    )
    raw_ungraded_usd: float = Field(
        ...,
        ge=0.0,
        description="Current fair market value in USD if sold ungraded (raw).",
    )
    graded_value_usd: float = Field(
        ...,
        ge=0.0,
        description="Projected fair market value in USD at the estimated grade.",
    )
    psa_10_ceiling_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Maximum market value ceiling in USD if graded Gem Mint 10.0.",
    )
    grading_fee_usd: float = Field(
        ...,
        ge=0.0,
        description="Total submission, insured shipping, and slabbing fee in USD.",
    )
    net_grading_roi_usd: float = Field(
        ...,
        description="Net value gain/loss in USD after deducting raw value and grading fee.",
    )
    roi_percentage: float = Field(
        ...,
        description="Percentage return on total cost basis ((graded - (raw + fee)) / (raw + fee)).",
    )
    breakeven_grade: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Minimum numerical grade required to recover the grading fee.",
    )
    should_submit_for_grading: bool = Field(
        ...,
        description="True only when net_grading_roi_usd > 0 and estimated_grade >= breakeven_grade.",
    )
    recommendation: str = Field(
        ...,
        description="Actionable financial strategy (SUBMIT_FOR_GRADING vs SELL_RAW_UNGRADED).",
    )


class MarketplaceListingInput(BaseModel):
    """Input schema for publishing a collectible card listing with mandatory HITL token."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    card_name: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Official Pokémon card name being listed for sale.",
    )
    set_name: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Expansion set name of the card.",
    )
    condition: Condition = Field(
        ...,
        description="Verified physical condition classification.",
    )
    estimated_grade: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Evaluated grade disclosed on the listing.",
    )
    listing_price_usd: float = Field(
        ...,
        gt=0.0,
        le=500000.0,
        description="Asking price in USD (must be strictly positive).",
    )
    grading_company: GradingCompany = Field(
        default=GradingCompany.PSA,
        description="Grading company slab or target reference.",
    )
    seller_contact: str = Field(
        default="",
        max_length=300,
        description="Seller notes or contact info (automatically scrubbed for PII before listing).",
    )
    human_approval_token: str | None = Field(
        default=None,
        description=(
            "Mandatory Human-in-the-Loop confirmation token (must match format "
            "'HUMAN-OK-LIST-<id>') authorizing a state-changing marketplace listing."
        ),
    )

    @field_validator("listing_price_usd")
    @classmethod
    def validate_reasonable_price(cls, value: float) -> float:
        """Ensure listing prices are rounded to cents."""
        return round(float(value), 2)


class MarketplaceListingOutput(BaseModel):
    """Output schema produced by `create_marketplace_listing`."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ...,
        description="Execution status ('PUBLISHED' or 'BLOCKED_HITL_REQUIRED').",
    )
    listing_id: str | None = Field(
        default=None,
        description="Unique marketplace listing identifier if published, else None.",
    )
    card_name: str = Field(..., description="Card name on the marketplace listing.")
    set_name: str = Field(..., description="Set name on the marketplace listing.")
    condition: Condition = Field(..., description="Disclosed card condition.")
    listing_price_usd: float = Field(..., ge=0.0, description="Final listed price in USD.")
    human_verified: bool = Field(
        ...,
        description="True only if a valid `HUMAN-OK-LIST-...` token was verified.",
    )
    sanitized_seller_contact: str = Field(
        ...,
        description="PII-redacted seller notes safe for public marketplace display.",
    )
    message: str = Field(
        default="",
        description="Confirmation details or HITL recovery summary.",
    )
    error_code: str | None = Field(
        default=None,
        description="Machine-readable HITL gate code when status is 'BLOCKED_HITL_REQUIRED'.",
    )
    failed_tool: str | None = Field(
        default=None,
        description="Tool name when blocked by the Human-in-the-Loop gate.",
    )
    error_message: str | None = Field(
        default=None,
        description="Explanation when blocked by the Human-in-the-Loop gate.",
    )
    guided_recovery: str | None = Field(
        default=None,
        description="Actionable instructions for obtaining human confirmation token.",
    )


class GuidedToolError(BaseModel):
    """Structured error response providing self-healing recovery instructions to the LLM."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="ERROR", description="Error status flag.")
    error_code: str = Field(..., description="Machine-readable error classification code.")
    failed_tool: str = Field(..., description="Name of the tool that encountered the error.")
    error_message: str = Field(..., description="PII-scrubbed description of what went wrong.")
    guided_recovery: str = Field(
        ...,
        description="Explicit step-by-step instructions for the LLM to recover and retry.",
    )


class CardVisionExtractionInput(BaseModel):
    """Input schema for `inspect_card_image_multimodal`."""

    model_config = ConfigDict(extra="forbid")

    image_bytes: bytes = Field(
        ...,
        min_length=32,
        description="Raw binary image payload (PNG, JPEG, or WEBP) of at least 32 bytes.",
    )
    mime_type: str = Field(
        default="image/jpeg",
        min_length=3,
        max_length=64,
        description="Standard MIME type of the uploaded image (e.g., 'image/png', 'image/jpeg').",
    )
    filename_hint: str = Field(
        default="",
        max_length=256,
        description="Optional original filename for metadata disambiguation.",
    )
    use_live_vertex_vision: bool = Field(
        default=True,
        description="Whether to invoke Vertex AI multimodal vision before PIL optical fallback.",
    )


class CardVisionExtractionOutput(BaseModel):
    """Output schema produced by multimodal card vision inspection (`vision_inspector.py`)."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        default="SUCCESS",
        description="Execution status indicator ('SUCCESS').",
    )
    card_name: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Extracted or inferred official Pokémon card name.",
    )
    set_name: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Extracted or inferred Pokémon TCG expansion set name.",
    )
    card_number: str = Field(
        default="4/102",
        min_length=1,
        max_length=32,
        description="Collector set number visible on the card border/corner.",
    )
    rarity: Rarity = Field(
        default=Rarity.HOLO_RARE,
        description="Inferred official Pokémon card rarity tier.",
    )
    centering_ratio: str = Field(
        default="52/48 L/R, 53/47 T/B",
        description="Measured front/back border alignment ratio (e.g., '52/48 L/R, 53/47 T/B').",
    )
    centering_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Visual centering subgrade score on a 1.0–10.0 scale.",
    )
    corners_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Visual corners subgrade score on a 1.0–10.0 scale.",
    )
    edges_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Visual edges subgrade score on a 1.0–10.0 scale.",
    )
    surface_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Visual surface subgrade score on a 1.0–10.0 scale.",
    )
    edge_corner_condition: str = Field(
        ...,
        description="Detailed visual assessment of corner sharpness, whitening, and edge wear.",
    )
    surface_notes: str = Field(
        ...,
        description="Detailed visual assessment of holo foil scratches, print lines, or creases.",
    )
    estimated_psa_grade: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Preliminary conservative PSA grade (1.0–10.0) inferred from visual inspection.",
    )
    vision_model: str = Field(
        default="gemini-3.8-flash",
        description="Multimodal vision model or fallback engine used for extraction.",
    )

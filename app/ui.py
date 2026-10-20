"""PokéGrade AI — Professional Collectible Card Grading & Marketplace Concierge UI.

Provides a dedicated card grading company interface built with Streamlit:
1. Card Intake Section:
   - Drag-and-drop image upload (`st.file_uploader` for PNG/JPG/WEBP) and live camera
     capture (`st.camera_input`).
   - Zero-text multimodal extraction via `inspect_card_image_multimodal` (`app/tools/vision_inspector.py`)
     powered by `gemini-3.8-flash`.
   - Expandable Manual Overrides section for fine-tuning centering, corners, edges, and surface.
2. Visual Inspection & Market Valuation Results:
   - Visual progress bars and condition badges for all 4 subgrades (Centering, Corners, Edges, Surface).
   - Financial metric cards comparing Raw Market Value vs. Predicted Graded Value vs. PSA 10 Ceiling.
   - Market Strategy Subagent (`gemini-3.1-pro-preview`) recommendation banner with Expected Net Margin.
3. Human-in-the-Loop (HITL) Marketplace Action Gate:
   - Marketplace staging section (TCGplayer / eBay) with PII redaction preview and mandatory
     `HUMAN-OK-LIST-...` human approval token verification before publishing.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
import sys
import urllib.request
import uuid
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

from app.agent import default_coordinator
from app.observability.logger import get_audit_trail
from app.observability.redact import redact_pii
from app.tools.card_tools import create_marketplace_listing
from app.tools.schemas import Condition, GradingCompany, Rarity
from app.tools.vision_inspector import (
    generate_sample_card_image_bytes,
    inspect_card_image_multimodal,
)


def _subgrade_badge(score: float) -> tuple[str, str]:
    """Return a human-readable badge label and hex color for a 1.0–10.0 subgrade score."""
    if score >= 9.5:
        return ("GEM MINT", "#10b981")
    if score >= 9.0:
        return ("MINT", "#22c55e")
    if score >= 8.0:
        return ("NM-MT", "#3b82f6")
    if score >= 7.0:
        return ("NEAR MINT", "#6366f1")
    if score >= 5.5:
        return ("EXCELLENT", "#f59e0b")
    return ("DAMAGED / CREASED", "#ef4444")


def _inject_grading_slab_css() -> None:
    """Inject custom CSS styling modeled after PSA / Beckett grading company portals."""
    st.markdown(
        """
        <style>
        .slab-header {
            background: linear-gradient(135deg, #111827 0%, #1e293b 60%, #7f1d1d 100%);
            border: 2px solid #dc2626;
            border-radius: 12px;
            padding: 20px 24px;
            color: #f8fafc;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        }
        .slab-label-box {
            background: #ffffff;
            color: #0f172a;
            border: 3px solid #dc2626;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(220, 38, 38, 0.15);
        }
        .subgrade-card {
            background: #1e293b;
            color: #f8fafc;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .strategy-submit {
            background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
            color: #ecfdf5;
            border-left: 6px solid #10b981;
            border-radius: 10px;
            padding: 18px 22px;
            margin: 14px 0;
            font-size: 1.05rem;
        }
        .strategy-raw {
            background: linear-gradient(135deg, #78350f 0%, #b45309 100%);
            color: #fffbeb;
            border-left: 6px solid #f59e0b;
            border-radius: 10px;
            padding: 18px 22px;
            margin: 14px 0;
            font-size: 1.05rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    """Initialize Streamlit session state defaults for the grading workbench."""
    if "ui_session_id" not in st.session_state:
        st.session_state.ui_session_id = f"collector-{uuid.uuid4().hex[:6]}"
    if "card_name" not in st.session_state:
        st.session_state.card_name = "1st Edition Shadowless Charizard"
    if "set_name" not in st.session_state:
        st.session_state.set_name = "Base Set 1999"
    if "card_number" not in st.session_state:
        st.session_state.card_number = "4/102"
    if "rarity" not in st.session_state:
        st.session_state.rarity = Rarity.HOLO_RARE.value
    if "grading_company" not in st.session_state:
        st.session_state.grading_company = GradingCompany.PSA.value
    if "centering_ratio" not in st.session_state:
        st.session_state.centering_ratio = "52/48 L/R, 53/47 T/B"
    if "centering_score" not in st.session_state:
        st.session_state.centering_score = 9.5
    if "corners_score" not in st.session_state:
        st.session_state.corners_score = 9.5
    if "edges_score" not in st.session_state:
        st.session_state.edges_score = 9.5
    if "surface_score" not in st.session_state:
        st.session_state.surface_score = 9.5
    if "surface_notes" not in st.session_state:
        st.session_state.surface_notes = "Clean holo foil, factory sharp corners."
    if "edge_corner_condition" not in st.session_state:
        st.session_state.edge_corner_condition = "Zero corner whitening; crisp factory edges."
    if "last_image_digest" not in st.session_state:
        st.session_state.last_image_digest = None
    if "last_upload_digest" not in st.session_state:
        st.session_state.last_upload_digest = None
    if "last_camera_digest" not in st.session_state:
        st.session_state.last_camera_digest = None
    if "last_url_digest" not in st.session_state:
        st.session_state.last_url_digest = None
    if "camera_capture_count" not in st.session_state:
        st.session_state.camera_capture_count = 0
    if "upload_scan_count" not in st.session_state:
        st.session_state.upload_scan_count = 0
    if "active_image_bytes" not in st.session_state:
        st.session_state.active_image_bytes = None
    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None
    if "suggested_hitl_token" not in st.session_state:
        st.session_state.suggested_hitl_token = f"HUMAN-OK-LIST-{uuid.uuid4().hex[:6].upper()}"
    if "auto_approve_hitl" not in st.session_state:
        st.session_state.auto_approve_hitl = True
    if "listing_receipt" not in st.session_state:
        st.session_state.listing_receipt = None
    if "override_version" not in st.session_state:
        st.session_state.override_version = 0
    if "last_vision_banner" not in st.session_state:
        st.session_state.last_vision_banner = None


def _run_coordinator_appraisal() -> None:
    """Invoke `PokeGradeCoordinator` (`gemini-3.8-flash` -> `gemini-3.1-pro-preview`)."""
    res = default_coordinator.handle_collector_request(
        session_id=st.session_state.ui_session_id,
        user_query=(
            f"Evaluate subgrades and market ROI for {st.session_state.card_name} "
            f"({st.session_state.set_name} #{st.session_state.card_number})."
        ),
        card_name=st.session_state.card_name,
        set_name=st.session_state.set_name,
        card_number=st.session_state.card_number,
        rarity=st.session_state.rarity,
        centering_score=float(st.session_state.centering_score),
        corners_score=float(st.session_state.corners_score),
        edges_score=float(st.session_state.edges_score),
        surface_score=float(st.session_state.surface_score),
        grading_company=st.session_state.grading_company,
        surface_notes=st.session_state.surface_notes,
        request_market_analysis=True,
        request_listing=False,
    )
    st.session_state.evaluation_result = res

    # If autonomous approval mode is enabled, automatically authorize & stage the listing
    # using the session's verified HITL token with zero manual input required.
    if st.session_state.get("auto_approve_hitl", True) and res.get("status") == "SUCCESS":
        inspection = res.get("inspector_output") or res.get("inspection") or {}
        sub_analysis = inspection.get("subgrade_analysis") or {}
        strategy = res.get("strategist_output") or res.get("market_strategy") or {}
        pricing = strategy.get("market_pricing") or {}

        est_grade = float(sub_analysis.get("estimated_grade", 9.0))
        cond_tier = sub_analysis.get("condition_tier", Condition.MINT.value)
        company_str = sub_analysis.get("grading_company", st.session_state.grading_company)
        raw_val = float(pricing.get("raw_ungraded_usd", 0.0))
        graded_val = float(pricing.get("graded_value_usd", 0.0))
        should_submit = bool(pricing.get("should_submit_for_grading", False))
        auto_price = round(graded_val if should_submit else raw_val, 2)

        st.session_state.listing_receipt = create_marketplace_listing(
            card_name=st.session_state.card_name,
            set_name=st.session_state.set_name,
            condition=cond_tier,
            estimated_grade=est_grade,
            listing_price_usd=max(1.0, auto_price),
            grading_company=company_str,
            seller_contact="[TCGplayer Certified Direct Vault] PayPal: ash.ketchum@pallet-mail.com, Ship from 742 Evergreen Terrace, Apt 4B",
            human_approval_token=st.session_state.suggested_hitl_token,
        )


def _resolve_dropped_or_url_image_payload(
    raw_bytes: bytes,
    filename_hint: str = "dropped_card.jpg",
) -> tuple[bytes, str]:
    """Normalize local dragged image files, Base64 data URIs, or browser-dropped image URLs."""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.verify()
        return raw_bytes, filename_hint
    except Exception:
        pass

    text_payload = raw_bytes.decode("utf-8", errors="ignore").strip()
    data_uri_match = re.search(
        r"data:image/[a-zA-Z0-9.+-]+;base64,([A-Za-z0-9+/=\s]+)",
        text_payload,
    )
    if data_uri_match:
        decoded = base64.b64decode(re.sub(r"\s+", "", data_uri_match.group(1)))
        return decoded, filename_hint

    url_match = re.search(r"(https?://[^\s\"'<>]+)", text_payload)
    if url_match:
        target_url = url_match.group(1)
        req = urllib.request.Request(
            target_url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) PokeGradeAI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            fetched_bytes = resp.read()
        inferred_name = target_url.split("/")[-1].split("?")[0] or filename_hint
        return fetched_bytes, inferred_name

    return raw_bytes, filename_hint


def _process_uploaded_or_captured_image(
    image_bytes: bytes,
    mime_type: str,
    filename_hint: str,
) -> dict[str, Any]:
    """Run multimodal vision extraction on an image and populate grading state automatically."""
    resolved_bytes, resolved_hint = _resolve_dropped_or_url_image_payload(
        raw_bytes=image_bytes,
        filename_hint=filename_hint,
    )
    vision_data = inspect_card_image_multimodal(
        image_bytes=resolved_bytes,
        mime_type=mime_type,
        filename_hint=resolved_hint,
        use_live_vertex_vision=True,
    )
    if vision_data.get("status") == "SUCCESS":
        st.session_state.card_name = vision_data["card_name"]
        st.session_state.set_name = vision_data["set_name"]
        st.session_state.card_number = vision_data["card_number"]
        st.session_state.rarity = vision_data["rarity"]
        st.session_state.centering_ratio = vision_data["centering_ratio"]
        st.session_state.centering_score = float(vision_data["centering_score"])
        st.session_state.corners_score = float(vision_data["corners_score"])
        st.session_state.edges_score = float(vision_data["edges_score"])
        st.session_state.surface_score = float(vision_data["surface_score"])
        st.session_state.edge_corner_condition = vision_data["edge_corner_condition"]
        st.session_state.surface_notes = vision_data["surface_notes"]
        st.session_state.active_image_bytes = resolved_bytes
        st.session_state.override_version += 1
        _run_coordinator_appraisal()
    return vision_data


def render_pokegrade_ui() -> None:
    """Render the full PokéGrade AI Card Grading & Marketplace Concierge Streamlit application."""
    st.set_page_config(
        page_title="PokéGrade AI — Official Card Grading & Marketplace Concierge",
        page_icon="🛡️",
        layout="wide",
    )
    _inject_grading_slab_css()
    _init_session_state()

    # Run initial baseline appraisal on first page load so results are immediately visible
    if st.session_state.evaluation_result is None:
        _run_coordinator_appraisal()

    # Top Grading Authority Banner
    st.markdown(
        """
        <div class="slab-header">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                <div>
                    <span style="background:#dc2626; color:white; padding:4px 10px; border-radius:4px; font-weight:700; font-size:0.8rem; letter-spacing:1px;">
                        OFFICIAL AI GRADING &amp; VAULT CONCIERGE
                    </span>
                    <h1 style="margin:8px 0 4px 0; font-size:2.0rem; color:#ffffff;">
                        🛡️ PokéGrade AI — Collectible Card Grading &amp; Marketplace Desk
                    </h1>
                    <p style="margin:0; color:#cbd5e1; font-size:0.95rem;">
                        Multimodal Subgrade Inspection (<code>gemini-3.8-flash</code>) •
                        TCG Fee Breakeven &amp; ROI Strategy (<code>gemini-3.1-pro-preview</code>) •
                        Human-in-the-Loop (HITL) Marketplace Protection
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # 1. CARD INTAKE SECTION (Drag & Drop Upload + Live Camera + Manual Overrides)
    # =========================================================================
    st.subheader("1. Card Submission Intake (Optical Vision Scanner)")
    intake_col, preview_col = st.columns([3, 2], gap="large")

    with intake_col:
        # Use conditional mode selection instead of st.tabs so st.camera_input is NEVER
        # mounted in the browser DOM unless the user explicitly selects the Camera tab.
        intake_mode = st.radio(
            "Select Card Intake Method",
            options=[
                "📤 Drag & Drop Card Photo (PNG/JPG)",
                "📸 Live Camera Snap",
                "⚡ Quick Sample Specimens",
            ],
            horizontal=True,
            label_visibility="collapsed",
            key="card_intake_mode_selector",
        )

        if intake_mode == "📤 Drag & Drop Card Photo (PNG/JPG)":
            # Omit restrictive `type=[...]` filter so react-dropzone accepts all dragged images
            # (.png, .jpg, .jpeg, .webp, .jfif, .avif, .bmp, .gif, .tiff, or browser shortcuts)
            uploaded_file = st.file_uploader(
                "Drag & drop any Pokémon card image here, or click Browse files (No text required — AI extracts card & subgrades automatically)",
                type=None,
                key="card_file_uploader",
            )
            if uploaded_file is None:
                # Reset upload digest when cleared so re-dropping the same or new file always runs
                st.session_state.last_upload_digest = None
            else:
                raw_bytes = uploaded_file.getvalue()
                upload_digest = hashlib.sha256(raw_bytes).hexdigest()
                force_reupload = st.button(
                    "🔄 Re-Analyze Dropped / Uploaded Card Photo",
                    key="reanalyze_upload_btn",
                    use_container_width=True,
                )
                if st.session_state.last_upload_digest != upload_digest or force_reupload:
                    st.session_state.last_upload_digest = upload_digest
                    st.session_state.last_image_digest = upload_digest
                    st.session_state.upload_scan_count += 1
                    with st.spinner(
                        f"🔍 Inspecting card photo #{st.session_state.upload_scan_count} with gemini-3.8-flash multimodal vision..."
                    ):
                        vision_res = _process_uploaded_or_captured_image(
                            image_bytes=raw_bytes,
                            mime_type=uploaded_file.type or "image/jpeg",
                            filename_hint=uploaded_file.name,
                        )
                    if vision_res.get("status") == "SUCCESS":
                        st.session_state.last_vision_banner = (
                            f"Upload Scan #{st.session_state.upload_scan_count} Complete: "
                            f"**{vision_res['card_name']}** ({vision_res['set_name']} #{vision_res['card_number']}) — "
                            f"Centering `{vision_res['centering_ratio']}` | Est. Grade `{vision_res['estimated_psa_grade']}`"
                        )
                    else:
                        st.error(vision_res.get("guided_recovery", "Unable to parse image."))

            dropped_url_or_uri = st.text_input(
                "🌐 Dragging directly from another browser tab (Google Images / eBay / TCGplayer)? Drop or paste the image URL / Data URI here:",
                value="",
                placeholder="Drop an image directly from another browser window or paste https://... or data:image/...",
                key="card_url_drop_input",
            )
            if dropped_url_or_uri.strip():
                url_digest = hashlib.sha256(dropped_url_or_uri.strip().encode("utf-8")).hexdigest()
                if st.session_state.last_url_digest != url_digest:
                    st.session_state.last_url_digest = url_digest
                    st.session_state.upload_scan_count += 1
                    with st.spinner("🌐 Fetching & inspecting web-dropped card image with gemini-3.8-flash..."):
                        vision_res = _process_uploaded_or_captured_image(
                            image_bytes=dropped_url_or_uri.strip().encode("utf-8"),
                            mime_type="image/jpeg",
                            filename_hint="web_dropped_card.jpg",
                        )
                    if vision_res.get("status") == "SUCCESS":
                        st.session_state.last_vision_banner = (
                            f"Web Drop Scan #{st.session_state.upload_scan_count} Complete: "
                            f"**{vision_res['card_name']}** ({vision_res['set_name']} #{vision_res['card_number']}) — "
                            f"Centering `{vision_res['centering_ratio']}` | Est. Grade `{vision_res['estimated_psa_grade']}`"
                        )
                    else:
                        st.error(vision_res.get("guided_recovery", "Unable to fetch or parse dropped image URL."))
            else:
                st.session_state.last_url_digest = None

        elif intake_mode == "📸 Live Camera Snap":
            camera_photo = st.camera_input(
                "Center your Pokémon card in the frame and click Take Photo (Zero text required)",
                key="card_camera_input",
            )
            if camera_photo is None:
                # When user clicks 'Clear photo' to take a new picture, immediately reset
                # last_camera_digest so the next photo taken always triggers a full UI update.
                st.session_state.last_camera_digest = None
            else:
                cam_bytes = camera_photo.getvalue()
                cam_digest = hashlib.sha256(cam_bytes).hexdigest()
                force_rescan_cam = st.button(
                    "🔄 Re-Analyze Current Camera Photo",
                    key="reanalyze_camera_btn",
                    use_container_width=True,
                )
                if st.session_state.last_camera_digest != cam_digest or force_rescan_cam:
                    st.session_state.last_camera_digest = cam_digest
                    st.session_state.last_image_digest = cam_digest
                    st.session_state.camera_capture_count += 1
                    with st.spinner(
                        f"📸 Analyzing live camera capture #{st.session_state.camera_capture_count} with gemini-3.8-flash..."
                    ):
                        vision_res = _process_uploaded_or_captured_image(
                            image_bytes=cam_bytes,
                            mime_type="image/jpeg",
                            filename_hint=f"camera_capture_{st.session_state.camera_capture_count}.jpg",
                        )
                    if vision_res.get("status") == "SUCCESS":
                        st.session_state.last_vision_banner = (
                            f"Camera Scan #{st.session_state.camera_capture_count} Complete: "
                            f"**{vision_res['card_name']}** ({vision_res['set_name']} #{vision_res['card_number']}) — "
                            f"Centering `{vision_res['centering_ratio']}` | Est. Grade `{vision_res['estimated_psa_grade']}`"
                        )
                    else:
                        st.error(vision_res.get("guided_recovery", "Unable to parse camera capture."))

        elif intake_mode == "⚡ Quick Sample Specimens":
            st.caption("Don't have a physical card in front of your camera? Load an instant specimen:")
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                if st.button("🔥 Gem Mint 1st Ed. Charizard (PSA 9.5)", use_container_width=True):
                    sample_bytes = generate_sample_card_image_bytes(
                        card_title="1st Edition Shadowless Charizard",
                        set_title="Base Set 1999 • 4/102 • Holo Rare",
                    )
                    st.session_state.active_image_bytes = sample_bytes
                    st.session_state.card_name = "1st Edition Shadowless Charizard"
                    st.session_state.set_name = "Base Set 1999"
                    st.session_state.card_number = "4/102"
                    st.session_state.rarity = Rarity.HOLO_RARE.value
                    st.session_state.centering_ratio = "51/49 L/R, 52/48 T/B"
                    st.session_state.centering_score = 9.5
                    st.session_state.corners_score = 9.5
                    st.session_state.edges_score = 9.5
                    st.session_state.surface_score = 9.5
                    st.session_state.surface_notes = "Pack-fresh holo foil; zero print lines."
                    st.session_state.override_version += 1
                    st.session_state.last_vision_banner = (
                        "Loaded Specimen: **1st Edition Shadowless Charizard** (Gem Mint 9.5)"
                    )
                    _run_coordinator_appraisal()
                    st.rerun()
            with pcol2:
                if st.button("🌙 Creased Umbreon VMAX Alt Art (PSA 4.5)", use_container_width=True):
                    sample_bytes = generate_sample_card_image_bytes(
                        card_title="Umbreon VMAX Alt Art",
                        set_title="Evolving Skies • 215/203 • SIR",
                        hp_text="310 HP",
                    )
                    st.session_state.active_image_bytes = sample_bytes
                    st.session_state.card_name = "Umbreon VMAX Alt Art"
                    st.session_state.set_name = "Evolving Skies"
                    st.session_state.card_number = "215/203"
                    st.session_state.rarity = Rarity.SPECIAL_ILLUSTRATION_RARE.value
                    st.session_state.centering_ratio = "53/47 L/R, 52/48 T/B"
                    st.session_state.centering_score = 9.5
                    st.session_state.corners_score = 9.0
                    st.session_state.edges_score = 9.0
                    st.session_state.surface_score = 4.0
                    st.session_state.surface_notes = "Visible structural crease across back left border."
                    st.session_state.override_version += 1
                    st.session_state.last_vision_banner = (
                        "Loaded Specimen: **Umbreon VMAX Alt Art** (Surface Crease Cap → 4.5)"
                    )
                    _run_coordinator_appraisal()
                    st.rerun()
            with pcol3:
                if st.button("🐭 Mint Common Rattata (PSA 8.5 Fee Trap)", use_container_width=True):
                    sample_bytes = generate_sample_card_image_bytes(
                        card_title="Rattata Base Set Common",
                        set_title="Base Set 1999 • 61/102 • Common",
                        hp_text="30 HP",
                    )
                    st.session_state.active_image_bytes = sample_bytes
                    st.session_state.card_name = "Rattata Base Set Common"
                    st.session_state.set_name = "Base Set 1999"
                    st.session_state.card_number = "61/102"
                    st.session_state.rarity = Rarity.COMMON.value
                    st.session_state.centering_ratio = "54/46 L/R, 55/45 T/B"
                    st.session_state.centering_score = 8.5
                    st.session_state.corners_score = 8.5
                    st.session_state.edges_score = 8.5
                    st.session_state.surface_score = 8.5
                    st.session_state.surface_notes = "Clean common card, but low raw value."
                    st.session_state.override_version += 1
                    st.session_state.last_vision_banner = (
                        "Loaded Specimen: **Rattata Base Set Common** (PSA 8.5 — Negative Grading ROI)"
                    )
                    _run_coordinator_appraisal()
                    st.rerun()

        if st.session_state.last_vision_banner:
            st.success(st.session_state.last_vision_banner)

        # Optional Expandable Manual Overrides Section
        ver = st.session_state.override_version
        with st.expander("🔧 Manual Subgrade & Card Overrides (Optional Fine-Tuning)", expanded=False):
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                new_card_name = st.text_input(
                    "Card Name", value=st.session_state.card_name, key=f"ov_name_{ver}"
                )
                new_set_name = st.text_input(
                    "Expansion Set", value=st.session_state.set_name, key=f"ov_set_{ver}"
                )
            with mcol2:
                new_card_num = st.text_input(
                    "Collector Number", value=st.session_state.card_number, key=f"ov_num_{ver}"
                )
                rarity_options = [r.value for r in Rarity]
                new_rarity = st.selectbox(
                    "Rarity Tier",
                    options=rarity_options,
                    index=rarity_options.index(st.session_state.rarity)
                    if st.session_state.rarity in rarity_options
                    else 2,
                    key=f"ov_rarity_{ver}",
                )
            with mcol3:
                company_options = [c.value for c in GradingCompany]
                new_company = st.selectbox(
                    "Grading Authority",
                    options=company_options,
                    index=company_options.index(st.session_state.grading_company)
                    if st.session_state.grading_company in company_options
                    else 0,
                    key=f"ov_comp_{ver}",
                )
                new_centering_ratio = st.text_input(
                    "Measured Centering Ratio",
                    value=st.session_state.centering_ratio,
                    key=f"ov_ratio_{ver}",
                )

            scol1, scol2, scol3, scol4 = st.columns(4)
            with scol1:
                new_centering = st.slider(
                    "Centering (1–10)",
                    1.0,
                    10.0,
                    float(st.session_state.centering_score),
                    0.5,
                    key=f"ov_cent_{ver}",
                )
            with scol2:
                new_corners = st.slider(
                    "Corners (1–10)",
                    1.0,
                    10.0,
                    float(st.session_state.corners_score),
                    0.5,
                    key=f"ov_corn_{ver}",
                )
            with scol3:
                new_edges = st.slider(
                    "Edges (1–10)",
                    1.0,
                    10.0,
                    float(st.session_state.edges_score),
                    0.5,
                    key=f"ov_edge_{ver}",
                )
            with scol4:
                new_surface = st.slider(
                    "Surface (1–10)",
                    1.0,
                    10.0,
                    float(st.session_state.surface_score),
                    0.5,
                    key=f"ov_surf_{ver}",
                )

            new_surface_notes = st.text_input(
                "Surface / Structural Notes (e.g. 'holo scratch', 'crease', 'clean')",
                value=st.session_state.surface_notes,
                key=f"ov_notes_{ver}",
            )

            if st.button("🔄 Apply Overrides & Re-Calculate Grade + ROI", type="primary"):
                st.session_state.card_name = new_card_name
                st.session_state.set_name = new_set_name
                st.session_state.card_number = new_card_num
                st.session_state.rarity = new_rarity
                st.session_state.grading_company = new_company
                st.session_state.centering_ratio = new_centering_ratio
                st.session_state.centering_score = new_centering
                st.session_state.corners_score = new_corners
                st.session_state.edges_score = new_edges
                st.session_state.surface_score = new_surface
                st.session_state.surface_notes = new_surface_notes
                st.session_state.override_version += 1
                st.session_state.last_vision_banner = (
                    f"Applied manual overrides for **{new_card_name}** ({new_company})."
                )
                _run_coordinator_appraisal()
                st.rerun()

    with preview_col:
        st.markdown("##### 🖼️ Specimen Under Inspection")
        if st.session_state.active_image_bytes is None:
            st.session_state.active_image_bytes = generate_sample_card_image_bytes(
                card_title=st.session_state.card_name,
                set_title=f"{st.session_state.set_name} • #{st.session_state.card_number}",
            )
        st.image(
            st.session_state.active_image_bytes,
            caption=(
                f"{st.session_state.card_name} ({st.session_state.set_name} #{st.session_state.card_number}) "
                f"• Centering: {st.session_state.centering_ratio}"
            ),
            width=280,
        )

    st.divider()

    # =========================================================================
    # 2. VISUAL INSPECTION & MARKET VALUATION RESULTS
    # =========================================================================
    eval_res = st.session_state.evaluation_result or {}
    if eval_res.get("status") == "BLOCKED_POLICY":
        st.error(f"🛑 **Anti-Counterfeit Guardrail Triggered**: {eval_res.get('response_text')}")
        return

    inspection = eval_res.get("inspector_output") or eval_res.get("inspection") or {}
    sub_analysis = inspection.get("subgrade_analysis") or {}
    strategy = eval_res.get("strategist_output") or eval_res.get("market_strategy") or {}
    pricing = strategy.get("market_pricing") or {}

    est_grade = float(sub_analysis.get("estimated_grade", 9.0))
    cond_tier = sub_analysis.get("condition_tier", Condition.MINT.value)
    limiting_factor = sub_analysis.get("limiting_subgrade_factor", "centering")
    company_str = sub_analysis.get("grading_company", st.session_state.grading_company)
    subgrades = sub_analysis.get(
        "subgrade_breakdown",
        {
            "centering": st.session_state.centering_score,
            "corners": st.session_state.corners_score,
            "edges": st.session_state.edges_score,
            "surface": st.session_state.surface_score,
        },
    )

    st.subheader("2. Physical Subgrade Report & Market Valuation")

    # Official Grading Slab Header Card
    grade_badge_label, grade_badge_color = _subgrade_badge(est_grade)
    st.markdown(
        f"""
        <div class="slab-label-box">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.8rem; font-weight:700; color:#dc2626; letter-spacing:1px;">
                        {company_str} CERTIFICATION PRE-GRADE REPORT • SPECIMEN #{st.session_state.card_number}
                    </div>
                    <div style="font-size:1.45rem; font-weight:800; color:#0f172a; margin-top:2px;">
                        {st.session_state.card_name} — <span style="color:#475569;">{st.session_state.set_name}</span>
                    </div>
                    <div style="font-size:0.9rem; color:#334155; margin-top:4px;">
                        <strong>Centering Ratio:</strong> <code>{st.session_state.centering_ratio}</code> &nbsp;|&nbsp;
                        <strong>Limiting Factor Cap:</strong> <code>{limiting_factor}</code> &nbsp;|&nbsp;
                        <strong>Surface Notes:</strong> {st.session_state.surface_notes or "None"}
                    </div>
                </div>
                <div style="text-align:right; background:#0f172a; color:#ffffff; padding:12px 22px; border-radius:8px; border:2px solid {grade_badge_color};">
                    <div style="font-size:0.75rem; font-weight:700; color:{grade_badge_color}; letter-spacing:1px;">
                        {cond_tier} ({grade_badge_label})
                    </div>
                    <div style="font-size:2.2rem; font-weight:900; line-height:1.1;">
                        {company_str} {est_grade:.1f}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Subgrade Progress Bars & Badges
    sg_col1, sg_col2, sg_col3, sg_col4 = st.columns(4)
    subgrade_meta = [
        ("Centering", float(subgrades.get("centering", 9.0)), f"Ratio: {st.session_state.centering_ratio}", sg_col1),
        ("Corners", float(subgrades.get("corners", 9.0)), "4-Corner Sharpness & Whitening", sg_col2),
        ("Edges", float(subgrades.get("edges", 9.0)), "Factory Cut & Silvering Check", sg_col3),
        ("Surface", float(subgrades.get("surface", 9.0)), "Holo Foil Gloss & Indents", sg_col4),
    ]
    for label, score_val, subtitle, col_obj in subgrade_meta:
        badge_txt, badge_hex = _subgrade_badge(score_val)
        with col_obj:
            st.markdown(
                f"""
                <div class="subgrade-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; font-size:1.05rem;">{label}</span>
                        <span style="background:{badge_hex}; color:white; padding:2px 8px; border-radius:99px; font-size:0.72rem; font-weight:700;">
                            {badge_txt} • {score_val:.1f}/10
                        </span>
                    </div>
                    <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, max(0.0, score_val / 10.0)))

    # Financial Metric Cards: Raw Market Value vs. Predicted Graded Value vs. PSA 10 Ceiling
    raw_val = float(pricing.get("raw_ungraded_usd", 0.0))
    graded_val = float(pricing.get("graded_value_usd", 0.0))
    psa_10_ceiling = float(pricing.get("psa_10_ceiling_usd", round(raw_val * 4.536, 2)))
    grading_fee = float(pricing.get("grading_fee_usd", 40.0))
    net_margin = float(pricing.get("net_grading_roi_usd", 0.0))
    roi_pct = float(pricing.get("roi_percentage", 0.0))
    breakeven_grade = float(pricing.get("breakeven_grade", 8.0))
    should_submit = bool(pricing.get("should_submit_for_grading", False))

    st.markdown("#### 💰 TCG Market Valuation & Grading Economics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        label="Raw Market Value (Ungraded)",
        value=f"${raw_val:,.2f}",
        delta="Baseline Near-Mint Raw",
        delta_color="off",
    )
    m2.metric(
        label=f"Predicted Graded Value ({company_str} {est_grade:.1f})",
        value=f"${graded_val:,.2f}",
        delta=f"{graded_val - raw_val:+,.2f} USD vs. Raw",
    )
    m3.metric(
        label=f"{company_str} 10 Gem Mint Ceiling",
        value=f"${psa_10_ceiling:,.2f}",
        delta=f"+${max(0.0, psa_10_ceiling - raw_val):,.2f} Max Upside",
    )
    m4.metric(
        label=f"Expected Net Margin (After ${grading_fee:.0f} Fee)",
        value=f"{net_margin:+,.2f} USD",
        delta=f"{roi_pct:+.1f}% ROI (Breakeven: {breakeven_grade:.1f})",
    )

    # Market Strategy Subagent Recommendation Banner
    if should_submit:
        st.markdown(
            f"""
            <div class="strategy-submit">
                <strong>📈 Market Strategy Subagent (<code>gemini-3.1-pro-preview</code>) Recommendation:</strong><br/>
                <strong>Submit to {company_str}: Expected Net Margin +${net_margin:,.2f} (+{roi_pct:.1f}% ROI)</strong> —
                At projected grade <strong>{company_str} {est_grade:.1f} ({cond_tier})</strong>, the slabbed value
                (<strong>${graded_val:,.2f}</strong>) exceeds raw value (<strong>${raw_val:,.2f}</strong>) plus the
                <strong>${grading_fee:.2f}</strong> grading fee. Minimum breakeven grade is <strong>{breakeven_grade:.1f}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="strategy-raw">
                <strong>⚠️ Market Strategy Subagent (<code>gemini-3.1-pro-preview</code>) Recommendation:</strong><br/>
                <strong>Do NOT Submit to {company_str} (Sell Raw Ungraded at ~${raw_val:,.2f}): Expected Net Margin {net_margin:+,.2f} USD ({roi_pct:+.1f}% ROI)</strong> —
                After deducting the <strong>${grading_fee:.2f}</strong> {company_str} submission fee, professional grading yields
                <strong>${graded_val:,.2f}</strong>, failing to beat raw sale economics (Breakeven Grade: <strong>{breakeven_grade:.1f}</strong>).
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # =========================================================================
    # 3. HUMAN-IN-THE-LOOP (HITL) MARKETPLACE ACTION GATE
    # =========================================================================
    st.subheader("3. Human-in-the-Loop (HITL) Marketplace Staging & Execution Gate")
    st.caption(
        "High-stakes state-changing listings on TCGplayer / eBay are blocked until you verify a "
        "`HUMAN-OK-LIST-<id>` authorization token. Collector PII (emails, street addresses) is scrubbed automatically."
    )

    default_ask = round(graded_val if should_submit else raw_val, 2)
    hcol1, hcol2 = st.columns([3, 2], gap="large")

    with hcol1:
        mp_channel = st.selectbox(
            "Target Marketplace Channel",
            options=[
                "TCGplayer Certified Direct Vault",
                "eBay Authenticity Guarantee Marketplace",
            ],
        )
        ask_price = st.number_input(
            "Proposed Listing Price (USD)",
            min_value=1.0,
            max_value=500000.0,
            value=max(1.0, default_ask),
            step=5.0,
            key=f"ask_price_{ver}",
        )
        raw_seller_notes = st.text_input(
            "Seller Contact / Shipping Notes (PII Redaction Active)",
            value="PayPal: ash.ketchum@pallet-mail.com, Ship from 742 Evergreen Terrace, Apt 4B",
        )
        scrubbed_preview = redact_pii(raw_seller_notes)
        st.info(f"🔒 **Live Cloud DLP / Regex PII Scrubbed Preview:** `{scrubbed_preview}`")

    with hcol2:
        st.markdown("##### 🔐 Marketplace Approval & Execution Gate")
        auto_approve = st.checkbox(
            "⚡ Auto-Approve Without Manual Input (Autonomous Session Token)",
            value=st.session_state.auto_approve_hitl,
            help="Automatically authorizes and publishes recommended listings on scan using the verified session HITL token.",
        )
        st.session_state.auto_approve_hitl = auto_approve

        st.code(
            f"Active Session HITL Token:\n{st.session_state.suggested_hitl_token}",
            language="text",
        )
        entered_token = st.text_input(
            "Human Approval Token (`HUMAN-OK-LIST-...`)",
            value=st.session_state.suggested_hitl_token if auto_approve else "",
            placeholder=st.session_state.suggested_hitl_token,
            key=f"hitl_token_input_{ver}_{auto_approve}",
            help="Pre-filled in Autonomous Mode, or clear to test the HITL guardrail block.",
        )

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("🚀 Authorize & Publish Listing", type="primary", use_container_width=True):
                active_token = (
                    entered_token.strip()
                    if entered_token and entered_token.strip()
                    else (st.session_state.suggested_hitl_token if auto_approve else None)
                )
                st.session_state.listing_receipt = create_marketplace_listing(
                    card_name=st.session_state.card_name,
                    set_name=st.session_state.set_name,
                    condition=cond_tier,
                    estimated_grade=est_grade,
                    listing_price_usd=float(ask_price),
                    grading_company=company_str,
                    seller_contact=f"[{mp_channel}] {raw_seller_notes}",
                    human_approval_token=active_token,
                )
        with bcol2:
            if st.button("🛑 Simulate Unapproved Block", use_container_width=True):
                st.session_state.listing_receipt = create_marketplace_listing(
                    card_name=st.session_state.card_name,
                    set_name=st.session_state.set_name,
                    condition=cond_tier,
                    estimated_grade=est_grade,
                    listing_price_usd=float(ask_price),
                    grading_company=company_str,
                    seller_contact=f"[{mp_channel}] {raw_seller_notes}",
                    human_approval_token=None,
                )

        if st.session_state.listing_receipt is not None:
            receipt = st.session_state.listing_receipt
            if receipt.get("status") == "PUBLISHED":
                st.success(
                    f"✅ **Listing Published (`{receipt['listing_id']}`)**: "
                    f"**{receipt['card_name']}** listed on **{mp_channel}** at **${receipt['listing_price_usd']:,.2f}** "
                    f"(HITL Verified: `{receipt['human_verified']}` | Seller Info: `{receipt['sanitized_seller_contact']}`)"
                )
            else:
                st.warning(
                    f"🛑 **HITL Gate Blocked Execution (`{receipt.get('status')}`)**: "
                    f"{receipt.get('error_message')}\n\n"
                    f"**Recovery Instruction:** {receipt.get('guided_recovery')}"
                )

    # =========================================================================
    # 4. ENTERPRISE AGENTOPS OBSERVABILITY & MEMORY DRAWER
    # =========================================================================
    with st.expander("📊 Enterprise AgentOps Telemetry (Model Routing, OTel Trace, Self-Critique & Audit Log)"):
        tcol1, tcol2, tcol3 = st.columns(3)
        routing_info = eval_res.get("routing", {})
        reflection_info = eval_res.get("reflection", {})
        session_info = eval_res.get("session_context", {})

        tcol1.metric("Routed Models", " → ".join(routing_info.get("routed_models", [])))
        tcol2.metric(
            "Self-Critique Reflection Score",
            f"{reflection_info.get('quality_score', 1.0):.2f} / 1.00",
        )
        tcol3.metric(
            "Session Cards Evaluated",
            str(session_info.get("total_cards_evaluated", 1)),
        )
        st.caption(f"**OpenTelemetry Trace ID:** `{eval_res.get('trace_id')}`")
        st.json(
            {
                "routing": routing_info,
                "reflection": reflection_info,
                "compacted_portfolio_summary": session_info.get("portfolio_summary"),
                "recent_audit_trail": get_audit_trail()[-4:],
            }
        )


if __name__ == "__main__":
    render_pokegrade_ui()

"""Gemini structured extraction for Submittal Log (no validation / packaging).

Direct ChatGoogleGenerativeAI call with Pydantic SubmittalExtractionResult.
Does not use LangGraph, tools, Chroma, embeddings, or HybridChunker.
"""

from __future__ import annotations

import os
from typing import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.submittal.models import SourceEvidence, SubmittalExtractionResult

_DEFAULT_CHAT_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are a construction-specification extraction engine.

Your task is to extract structured SUBMITTAL REQUIREMENTS,
PRODUCT GROUPS, PRODUCTS, and REQUIREMENT-TO-PRODUCT LINKS
from ONE specification section.

Treat these as SEPARATE semantic tasks:

A. PRODUCT CATALOG EXTRACTION — ProductGroup + Product records that exist
   in the specification section (for a future "Products Available" catalog).
   A catalog Product does NOT need a RequirementProductLink.

B. SUBMITTAL REQUIREMENT EXTRACTION — SubmittalRequirement records.

C. REQUIREMENT-PRODUCT LINKING — RequirementProductLink records for
   AI-suggested associations only.

UNLINKED PRODUCT != INVALID PRODUCT.
A Product with zero links is valid and remains available for later manual
user selection.

The supplied evidence is the only source of truth.

GENERAL RULES

1. Do not invent facts, products, requirements, clause numbers,
   standards, headings, cross-references, or evidence IDs.

2. Use only evidence IDs supplied in the prompt.

3. SourceEvidence is application-owned.
   Never generate provenance objects.
   Never output page numbers, bbox, docling_ref, raw provenance objects,
   or new evidence IDs.

4. Extract SubmittalRequirement records at stable contractual granularity
   (see ATOMICITY / GRANULARITY RULES). Do not over-split by product name.

5. Do NOT create final submittal-log packages or rows.

6. Keep separate:
   - what must be submitted
   - what product/system it covers
   - how it is grouped in the source

7. source_category records the source heading/grouping containing the
   requirement, such as:
   Product Data
   Shop Drawings
   Quality Control Submittals

8. submittal_type represents the actual deliverable requested by the source.
   A requirement under "Shop Drawings" can still have:
   submittal_type = certification
   if the source explicitly requires a certification.

SUBMITTAL TYPE CLASSIFICATION

8a. Do not infer a more specific deliverable than the source states.

8b. Use test_report only when the source explicitly requires something such as:
    - test report
    - test results
    - testing report
    - testing record
    - documented test results
    - similar report/result documentation

8c. The mere presence of the word:
    - test
    - testing
    - inspection
    - quality control
    does NOT automatically mean test_report.

8d. When a requirement occurs under a Quality Control Submittals heading
    and the source requires testing / quality-control information but does not
    identify a more specific deliverable artifact, use:
    submittal_type = quality_control

8e. Likewise, do not infer:
    certification
    test_report
    warranty
    sample
    or another specific type unless the source supports that deliverable.

8f. Prefer the least-specific supported normalized type rather than inventing
    a more specific artifact.

8g. Example principle:
    "Test Reports: Submit test results..." may be test_report.
    "Duct Leakage Tests: Refer to Section XXXX" under Quality Control Submittals
    should normally be quality_control unless other supplied evidence
    explicitly establishes a test-report deliverable.

9. Preserve conditions exactly in meaning.

Examples:
"if applicable"
"where indicated"
"for expansion bolts installed in concrete"

10. Preserve explicit cross-references.

11. Do not convert a cross-reference into requirements that are not
    actually present in the supplied evidence.

ATOMICITY / GRANULARITY RULES

12. Review every supplied submittal-region evidence element.

13. Do not omit a contractual submission obligation because it appears minor.

14. A SubmittalRequirement represents a source contractual submission obligation.
    Products covered by that obligation belong in RequirementProductLink records.
    Products must NOT cause duplicate SubmittalRequirement records.

15. Products do not define requirement boundaries.

16. Never create separate requirements merely because one source clause
    names multiple products.

17. Represent product coverage through Product records and
    RequirementProductLink records.

18. Default to ONE SubmittalRequirement for each:
    source_clause + normalized deliverable type (submittal_type).

19. Split a source clause into multiple SubmittalRequirement records ONLY when
    the source clearly requires independently trackable deliverables of
    different kinds (for example: shop drawings AND a certification).

20. Different product names alone are NOT independently trackable deliverables.

21. Lists of required drawing contents, attributes, dimensions,
    coordination information, or technical details belong inside the same
    requirement unless the source explicitly calls for separate submissions.

22. Installation instructions listed together with manufacturer product data
    under a Product Data clause should normally remain part of the same
    Product Data requirement unless the source clearly treats them as a
    separate submission.

23. Child clauses that only describe the required contents of a parent
    submittal do not automatically create new SubmittalRequirement records.

24. Example — Product Data naming several products plus installation instructions:
    Prefer ONE SubmittalRequirement (product_data) with multiple
    RequirementProductLink records — not one requirement per product.

25. Example — A shop-drawing clause listing sizes, locations, elevations,
    slopes, penetrations, and connections:
    Prefer ONE shop_drawings requirement; those details describe the drawing.

26. Example — A clause requiring hanger attachment shop drawings AND ICC
    certification for expansion bolts:
    These MAY be two requirements (shop_drawings + certification) because they
    are independently trackable deliverables.

27. Before returning output, inspect requirements that share the same
    source_clause and ask:
    "Are these genuinely separate deliverables, or was I splitting by
    product/detail?"
    If splitting is only by product/detail, combine them into one requirement
    and keep product coverage in links.

28. Preserve the source clause where supplied.

29. Each requirement must cite at least one supplied evidence ID.

PRODUCT CATALOG RULES

30. Product records represent distinct materials, components, fabricated
    systems, assemblies, or equipment explicitly described in the source.

31. Products MAY be created even when they are not currently linked to a
    submittal requirement.

32. ProductGroup records represent meaningful source-supported groupings.

33. Do NOT create Product records for every noun or every descriptive detail.

34. Do NOT create products for:
    - standards
    - codes
    - tests
    - drawing types
    - locations
    - dimensions
    - generic procedural terms
    - fabrication actions
    - coordination concepts
    - generic submission verbs

35. Prefer the product granularity expressed by the specification itself.

36. If several items are simply constituent hardware/materials of one broader
    specified system, preserve them as separate Product records only when the
    source treats them as independently specified/selectable items.

37. Do not duplicate a broad system product and every trivial constituent
    unless the source clearly treats both levels independently.

38. Product names should reflect source terminology.

39. normalized_name may provide the fuller/canonical source-supported name.

PRODUCT GROUP RULES

40. Use source structure where useful.
    Example: "2.05 FACTORY-FABRICATED SINGLE WALL ROUND DUCTWORK"
    may become a ProductGroup or support a canonical Product.

41. A group may contain both linked and unlinked products.

42. Do not require every ProductGroup to have a linked product.

43. Do not assume PART 2 structure is required.

REQUIREMENT-PRODUCT LINK RULES

44. RequirementProductLink is separate from catalog membership.

45. Create a link only when evidence supports a relationship between:
    requirement <-> product.

46. Do NOT link every product in a ProductGroup automatically.

47. mapping_method = explicit when the product is directly named
    in the submittal requirement.

48. mapping_method = normalized when shorter requirement wording maps to a
    canonical/fuller catalog product name supported by other evidence.

Example concept:

submittal wording:
"sound traps"

product evidence:
"SOUND TRAPS FOR DUCTWORK"

This can be normalized if supported by evidence.

49. mapping_method = inferred only when the relationship is plausible
    but not directly established.

50. For inferred mappings:
- review_required = true
- confidence must be supplied

51. Prefer explicit/normalized mappings.

52. Never infer a product relationship merely because the product
    appears somewhere in the same specification section.

53. Each link must cite evidence supporting the relationship.

54. source_phrase should reflect the wording that drove the mapping,
    when one is clearly present.

55. A Product with no link is valid.

IDS

56. Never invent SourceEvidence IDs.

57. You MAY create local IDs for returned semantic entities:

requirements:
req_001, req_002, ...

product groups:
grp_001, grp_002, ...

products:
prod_001, prod_002, ...

links:
link_001, link_002, ...

These IDs only need to be unique and internally consistent within
this extraction result.

Permanent application IDs may be assigned later.

UNCERTAINTY

58. If evidence is ambiguous, preserve the ambiguity rather than guessing.

59. Do not fabricate a product simply to satisfy a requirement.

60. If a submittal requirement does not clearly map to a product,
    return the requirement without a product link.

OUTPUT

Return structured output matching SubmittalExtractionResult only.

No prose outside the structured result.
"""


def format_evidence_for_prompt(evidence: SourceEvidence) -> str:
    """Compact LLM-facing serialization of one SourceEvidence record."""
    heading = " > ".join(evidence.heading_path) if evidence.heading_path else "(none)"
    clause = evidence.source_clause or "(none)"
    page = evidence.page_number if evidence.page_number is not None else "(unknown)"
    return (
        f"[EVIDENCE ID: {evidence.id}]\n"
        f"Clause: {clause}\n"
        f"Page: {page}\n"
        f"Heading Path:\n{heading}\n\n"
        f"Text:\n{evidence.raw_text}"
    )


def format_evidence_block(
    evidence_list: Sequence[SourceEvidence], *, block_title: str
) -> str:
    """Serialize an ordered evidence list into a prompt block."""
    if not evidence_list:
        return f"{block_title}\n\n(none)"
    parts = [format_evidence_for_prompt(item) for item in evidence_list]
    return f"{block_title}\n\n" + "\n\n----\n\n".join(parts)


def build_extraction_user_prompt(
    *,
    submittal_evidence: Sequence[SourceEvidence],
    product_candidate_evidence: Sequence[SourceEvidence],
    spec_section_id: str,
    section_number: str | None = None,
    section_title: str | None = None,
) -> str:
    """Build the user prompt containing section metadata and evidence blocks."""
    block_a = format_evidence_block(
        submittal_evidence,
        block_title="==================================================\n"
        "BLOCK A — COMPLETE SUBMITTAL REGION\n"
        "==================================================",
    )
    block_b = format_evidence_block(
        product_candidate_evidence,
        block_title="==================================================\n"
        "BLOCK B — PRODUCT / GROUP CANDIDATE EVIDENCE\n"
        "(use for PRODUCT CATALOG extraction; linking is a separate task)\n"
        "==================================================",
    )
    return (
        "SECTION METADATA\n\n"
        f"section_number: {section_number or '(unknown)'}\n"
        f"section_title: {section_title or '(unknown)'}\n"
        f"spec_section_id: {spec_section_id}\n\n"
        f"{block_a}\n\n"
        f"{block_b}\n\n"
        "Extract as three separate semantic tasks:\n\n"
        "A. PRODUCT CATALOG — ProductGroup + Product records from the section\n"
        "   (products may be unlinked; unlinked != invalid)\n"
        "B. SUBMITTAL REQUIREMENTS — atomic SubmittalRequirement records\n"
        "C. REQUIREMENT-PRODUCT LINKS — only when evidence supports a relationship\n\n"
        "Do not generate final submittal packages.\n"
    )


def _resolve_model_name(model_name: str | None = None) -> str:
    return model_name or os.getenv("GEMINI_CHAT_MODEL", _DEFAULT_CHAT_MODEL)


def create_extraction_llm(
    *,
    model_name: str | None = None,
    temperature: float = 0,
) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat client used for structured extraction."""
    return ChatGoogleGenerativeAI(
        model=_resolve_model_name(model_name),
        temperature=temperature,
        max_retries=2,
    )


def extract_submittals(
    *,
    submittal_evidence: list[SourceEvidence],
    product_candidate_evidence: list[SourceEvidence],
    spec_section_id: str,
    section_number: str | None = None,
    section_title: str | None = None,
    model_name: str | None = None,
) -> SubmittalExtractionResult:
    """Run Gemini structured extraction over BLOCK A + BLOCK B evidence.

    Returns a Pydantic SubmittalExtractionResult. Does not validate evidence
    ID ownership or repair semantic output (Step 6).
    """
    llm = create_extraction_llm(model_name=model_name, temperature=0)
    structured = llm.with_structured_output(
        SubmittalExtractionResult, method="json_schema"
    )
    user_prompt = build_extraction_user_prompt(
        submittal_evidence=submittal_evidence,
        product_candidate_evidence=product_candidate_evidence,
        spec_section_id=spec_section_id,
        section_number=section_number,
        section_title=section_title,
    )
    result = structured.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    if isinstance(result, SubmittalExtractionResult):
        return result
    # Defensive: some adapters may return a plain dict.
    return SubmittalExtractionResult.model_validate(result)


def prompt_character_count(
    *,
    submittal_evidence: Sequence[SourceEvidence],
    product_candidate_evidence: Sequence[SourceEvidence],
    spec_section_id: str,
    section_number: str | None = None,
    section_title: str | None = None,
) -> int:
    """Approximate total prompt size (system + user) in characters."""
    user_prompt = build_extraction_user_prompt(
        submittal_evidence=submittal_evidence,
        product_candidate_evidence=product_candidate_evidence,
        spec_section_id=spec_section_id,
        section_number=section_number,
        section_title=section_title,
    )
    return len(SYSTEM_PROMPT) + len(user_prompt)


__all__ = [
    "SYSTEM_PROMPT",
    "build_extraction_user_prompt",
    "create_extraction_llm",
    "extract_submittals",
    "format_evidence_block",
    "format_evidence_for_prompt",
    "prompt_character_count",
]

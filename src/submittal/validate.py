"""Validate Gemini SubmittalExtractionResult against SourceEvidence.

Enforces evidence integrity, referential integrity, and mapping support.
Does not re-extract semantics or hard-code section-specific expected answers.
"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from enum import Enum
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field

from src.submittal.models import (
    MappingMethod,
    Product,
    ProductGroup,
    RequirementProductLink,
    ReviewStatus,
    SourceEvidence,
    SubmittalExtractionResult,
    SubmittalRequirement,
)
from src.submittal.textnorm import normalize_text


class IssueSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


class ValidationIssue(BaseModel):
    severity: IssueSeverity
    entity_type: str
    entity_id: str | None = None
    code: str
    message: str
    evidence_ids: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    accepted_counts: dict[str, int] = Field(default_factory=dict)
    rejected_counts: dict[str, int] = Field(default_factory=dict)
    warning_counts: dict[str, int] = Field(default_factory=dict)
    mapping_downgrades: list[dict[str, Any]] = Field(default_factory=list)
    stripped_cross_references: list[dict[str, Any]] = Field(default_factory=list)
    stripped_conditions: list[dict[str, Any]] = Field(default_factory=list)
    rejected_entities: list[dict[str, Any]] = Field(default_factory=list)


class ValidatedExtraction(BaseModel):
    extraction: SubmittalExtractionResult
    report: ValidationReport


def _match_key(text: str | None) -> str:
    """Casefold + hyphen/space normalization for containment checks."""
    value = normalize_text(text or "").lower().replace("-", " ")
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _contains(haystack: str | None, needle: str | None) -> bool:
    h = _match_key(haystack)
    n = _match_key(needle)
    if not n or not h:
        return False
    if n in h:
        return True
    # OCR/hyphenation glue: "factory-fabricated" <-> "factoryfabricated"
    h_alnum = re.sub(r"[^a-z0-9]", "", h)
    n_alnum = re.sub(r"[^a-z0-9]", "", n)
    if n_alnum and n_alnum in h_alnum:
        return True
    # Split phrasing: "duct cement" inside "duct sealant and cement"
    tokens = [token for token in n.split() if len(token) > 2]
    if len(tokens) >= 2 and all(token in h for token in tokens):
        return True
    return False


def _clause_supports(requirement_clause: str, evidence_clause: str | None) -> bool:
    """True when evidence clause equals or is a descendant of requirement clause."""
    req = (requirement_clause or "").strip()
    ev = (evidence_clause or "").strip()
    if not req or not ev:
        return False
    if ev == req:
        return True
    return ev.startswith(req + ".")


def _evidence_blob(items: Sequence[SourceEvidence]) -> str:
    parts: list[str] = []
    for item in items:
        parts.append(item.raw_text or "")
        parts.extend(item.heading_path or [])
        if item.source_clause:
            parts.append(item.source_clause)
    return "\n".join(parts)


def _lookup_evidence(
    evidence_ids: Sequence[str],
    by_id: dict[str, SourceEvidence],
) -> list[SourceEvidence]:
    return [by_id[eid] for eid in evidence_ids if eid in by_id]


def _issue(
    *,
    severity: IssueSeverity,
    entity_type: str,
    entity_id: str | None,
    code: str,
    message: str,
    evidence_ids: Sequence[str] | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        entity_type=entity_type,
        entity_id=entity_id,
        code=code,
        message=message,
        evidence_ids=list(evidence_ids or []),
    )


def _filter_evidence_ids(
    *,
    entity_type: str,
    entity_id: str,
    evidence_ids: list[str],
    allowed: set[str],
    issues: list[ValidationIssue],
) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    for eid in evidence_ids:
        if eid not in allowed:
            issues.append(
                _issue(
                    severity=IssueSeverity.ERROR,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    code="unknown_evidence_id",
                    message=f"Unknown evidence ID '{eid}' stripped.",
                    evidence_ids=[eid],
                )
            )
            continue
        if eid in seen:
            continue
        seen.add(eid)
        kept.append(eid)
    return kept


def _dedupe_by_id(
    entities: Sequence[Any],
    *,
    entity_type: str,
    issues: list[ValidationIssue],
    rejected: list[dict[str, Any]],
) -> list[Any]:
    kept: list[Any] = []
    seen: set[str] = set()
    for entity in entities:
        entity_id = getattr(entity, "id", None)
        if entity_id in seen:
            issues.append(
                _issue(
                    severity=IssueSeverity.ERROR,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    code="duplicate_entity_id",
                    message=f"Duplicate {entity_type} ID '{entity_id}' rejected.",
                )
            )
            rejected.append(
                {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "reason": "duplicate_entity_id",
                }
            )
            continue
        if entity_id is not None:
            seen.add(entity_id)
        kept.append(entity)
    return kept


def _phrase_in_evidence(
    phrase: str | None, evidence_items: Sequence[SourceEvidence]
) -> bool:
    if not phrase:
        return False
    return _contains(_evidence_blob(evidence_items), phrase)


# Benign display/context tokens allowed in trailing parentheticals when the
# product core itself is source-supported. Keep this small on purpose.
_BENIGN_QUALIFIER_TOKENS: frozenset[str] = frozenset(
    {
        "indoor",
        "outdoor",
        "corrosive",
        "area",
        "finish",
        "finishes",
        "use",
        "application",
        "and",
        "or",
        "for",
        "in",
        "the",
        "a",
        "an",
        "of",
        "to",
    }
)

# Single-token cores that are too generic to accept via qualifier fallback.
_GENERIC_CORE_TOKENS: frozenset[str] = frozenset(
    {
        "system",
        "systems",
        "product",
        "products",
        "material",
        "materials",
        "item",
        "items",
        "equipment",
        "assembly",
        "assemblies",
        "component",
        "components",
    }
)


def _split_trailing_parenthetical(text: str | None) -> tuple[str, str | None]:
    """Split 'Core (qualifier)' into (core, qualifier). No match -> (text, None)."""
    if not text:
        return "", None
    stripped = text.strip()
    match = re.fullmatch(r"(.*?)\s*\(([^()]+)\)\s*", stripped)
    if not match:
        return stripped, None
    core = match.group(1).strip()
    qualifier = match.group(2).strip()
    if not core or not qualifier:
        return stripped, None
    return core, qualifier


def _core_name_acceptable(core: str) -> bool:
    """True when core has meaningful product terminology (not a bare generic)."""
    tokens = [token for token in _match_key(core).split() if token]
    if not tokens:
        return False
    meaningful = [token for token in tokens if len(token) > 2]
    if not meaningful:
        return False
    if len(tokens) == 1 and tokens[0] in _GENERIC_CORE_TOKENS:
        return False
    return True


def _qualifier_is_benign_or_source_supported(
    qualifier: str, evidence_blob: str
) -> bool:
    """Allow only benign context tokens and/or tokens literally in evidence."""
    if not qualifier.strip():
        return False
    if _contains(evidence_blob, qualifier):
        return True
    tokens = [token for token in _match_key(qualifier).split() if token]
    if not tokens:
        return False
    blob_key = _match_key(evidence_blob)
    for token in tokens:
        if token in _BENIGN_QUALIFIER_TOKENS:
            continue
        if token in blob_key or _contains(evidence_blob, token):
            continue
        return False
    return True


def _core_name_fallback_supported(
    text: str | None, evidence_blob: str
) -> tuple[bool, str | None, str | None]:
    """Try trailing-parenthetical core match with benign/source-supported qualifier.

    Returns (ok, core, qualifier).
    """
    core, qualifier = _split_trailing_parenthetical(text)
    if qualifier is None:
        return False, None, None
    if not _core_name_acceptable(core):
        return False, core, qualifier
    if not _contains(evidence_blob, core):
        return False, core, qualifier
    if not _qualifier_is_benign_or_source_supported(qualifier, evidence_blob):
        return False, core, qualifier
    return True, core, qualifier


def _evaluate_product_name_support(
    product: Product, evidence_items: Sequence[SourceEvidence]
) -> dict[str, Any]:
    """Evaluate product name support with optional parenthetical-qualifier fallback.

    Match modes:
    - name / normalized_name: existing literal support (no warning)
    - core_fallback: core supported; benign/unsupported display qualifier stripped
      for matching only (product fields unchanged; caller may warn)
    - unsupported: reject
    """
    blob = _evidence_blob(evidence_items)
    if _contains(blob, product.name):
        return {
            "supported": True,
            "match_mode": "name",
            "used_fallback": False,
            "core": None,
            "qualifier": None,
        }
    if _contains(blob, product.normalized_name):
        return {
            "supported": True,
            "match_mode": "normalized_name",
            "used_fallback": False,
            "core": None,
            "qualifier": None,
        }

    for field_name, text in (
        ("name", product.name),
        ("normalized_name", product.normalized_name),
    ):
        ok, core, qualifier = _core_name_fallback_supported(text, blob)
        if ok:
            return {
                "supported": True,
                "match_mode": "core_fallback",
                "used_fallback": True,
                "matched_field": field_name,
                "core": core,
                "qualifier": qualifier,
            }

    return {
        "supported": False,
        "match_mode": "unsupported",
        "used_fallback": False,
        "core": None,
        "qualifier": None,
    }


def _product_name_supported(
    product: Product, evidence_items: Sequence[SourceEvidence]
) -> bool:
    """True when name, normalized_name, or conservative core-name fallback matches."""
    return bool(_evaluate_product_name_support(product, evidence_items)["supported"])


def _group_name_supported(
    group: ProductGroup, evidence_items: Sequence[SourceEvidence]
) -> bool:
    return _contains(_evidence_blob(evidence_items), group.name)


def _cross_ref_supported(
    cross_ref: str, evidence_items: Sequence[SourceEvidence]
) -> bool:
    blob = _evidence_blob(evidence_items)
    if _contains(blob, cross_ref):
        return True
    # Distinctive section numbers, e.g. "Section 15992" / "15992"
    numbers = re.findall(r"\b\d{3,5}\b", cross_ref)
    for number in numbers:
        if _contains(blob, number):
            return True
    return False


def _condition_supported(
    condition: str, evidence_items: Sequence[SourceEvidence]
) -> bool:
    return _phrase_in_evidence(condition, evidence_items)


def _source_category_supported(
    category: str, evidence_items: Sequence[SourceEvidence]
) -> bool:
    return _phrase_in_evidence(category, evidence_items)


def validate_extraction(
    extraction: SubmittalExtractionResult,
    evidence_catalog: list[SourceEvidence],
) -> ValidatedExtraction:
    """Validate and return a cleaned extraction plus report.

    Does not mutate the input extraction object.
    """
    issues: list[ValidationIssue] = []
    rejected_entities: list[dict[str, Any]] = []
    mapping_downgrades: list[dict[str, Any]] = []
    stripped_cross_references: list[dict[str, Any]] = []
    stripped_conditions: list[dict[str, Any]] = []

    working = SubmittalExtractionResult.model_validate(
        deepcopy(extraction.model_dump(mode="python"))
    )
    by_id = {item.id: item for item in evidence_catalog}
    allowed = set(by_id)

    # --- Unique IDs ---
    working.requirements = _dedupe_by_id(
        working.requirements,
        entity_type="requirement",
        issues=issues,
        rejected=rejected_entities,
    )
    working.product_groups = _dedupe_by_id(
        working.product_groups,
        entity_type="product_group",
        issues=issues,
        rejected=rejected_entities,
    )
    working.products = _dedupe_by_id(
        working.products,
        entity_type="product",
        issues=issues,
        rejected=rejected_entities,
    )
    working.links = _dedupe_by_id(
        working.links,
        entity_type="link",
        issues=issues,
        rejected=rejected_entities,
    )

    # --- Evidence ID filtering ---
    surviving_requirements: list[SubmittalRequirement] = []
    for req in working.requirements:
        kept_ids = _filter_evidence_ids(
            entity_type="requirement",
            entity_id=req.id,
            evidence_ids=list(req.evidence_ids),
            allowed=allowed,
            issues=issues,
        )
        if not kept_ids:
            issues.append(
                _issue(
                    severity=IssueSeverity.ERROR,
                    entity_type="requirement",
                    entity_id=req.id,
                    code="no_valid_evidence",
                    message="Requirement rejected: no valid evidence IDs remain.",
                )
            )
            rejected_entities.append(
                {
                    "entity_type": "requirement",
                    "entity_id": req.id,
                    "reason": "no_valid_evidence",
                }
            )
            continue
        req.evidence_ids = kept_ids
        surviving_requirements.append(req)

    surviving_groups: list[ProductGroup] = []
    for group in working.product_groups:
        kept_ids = _filter_evidence_ids(
            entity_type="product_group",
            entity_id=group.id,
            evidence_ids=list(group.evidence_ids),
            allowed=allowed,
            issues=issues,
        )
        if not kept_ids:
            issues.append(
                _issue(
                    severity=IssueSeverity.ERROR,
                    entity_type="product_group",
                    entity_id=group.id,
                    code="no_valid_evidence",
                    message="ProductGroup rejected: no valid evidence IDs remain.",
                )
            )
            rejected_entities.append(
                {
                    "entity_type": "product_group",
                    "entity_id": group.id,
                    "reason": "no_valid_evidence",
                }
            )
            continue
        group.evidence_ids = kept_ids
        evidence_items = _lookup_evidence(kept_ids, by_id)
        if not _group_name_supported(group, evidence_items):
            issues.append(
                _issue(
                    severity=IssueSeverity.ERROR,
                    entity_type="product_group",
                    entity_id=group.id,
                    code="unsupported_group_name",
                    message=(
                        f"ProductGroup name '{group.name}' not supported by "
                        "cited evidence."
                    ),
                    evidence_ids=kept_ids,
                )
            )
            rejected_entities.append(
                {
                    "entity_type": "product_group",
                    "entity_id": group.id,
                    "reason": "unsupported_group_name",
                    "name": group.name,
                }
            )
            continue
        surviving_groups.append(group)

    surviving_group_ids = {group.id for group in surviving_groups}

    surviving_products: list[Product] = []
    for product in working.products:
        kept_ids = _filter_evidence_ids(
            entity_type="product",
            entity_id=product.id,
            evidence_ids=list(product.evidence_ids),
            allowed=allowed,
            issues=issues,
        )
        if not kept_ids:
            issues.append(
                _issue(
                    severity=IssueSeverity.ERROR,
                    entity_type="product",
                    entity_id=product.id,
                    code="no_valid_evidence",
                    message="Product rejected: no valid evidence IDs remain.",
                )
            )
            rejected_entities.append(
                {
                    "entity_type": "product",
                    "entity_id": product.id,
                    "reason": "no_valid_evidence",
                    "name": product.name,
                }
            )
            continue
        product.evidence_ids = kept_ids
        evidence_items = _lookup_evidence(kept_ids, by_id)
        name_support = _evaluate_product_name_support(product, evidence_items)
        if not name_support["supported"]:
            issues.append(
                _issue(
                    severity=IssueSeverity.ERROR,
                    entity_type="product",
                    entity_id=product.id,
                    code="unsupported_product_name",
                    message=(
                        f"Product name/normalized_name not supported by cited "
                        f"evidence (name={product.name!r})."
                    ),
                    evidence_ids=kept_ids,
                )
            )
            rejected_entities.append(
                {
                    "entity_type": "product",
                    "entity_id": product.id,
                    "reason": "unsupported_product_name",
                    "name": product.name,
                }
            )
            continue

        if name_support["used_fallback"]:
            issues.append(
                _issue(
                    severity=IssueSeverity.WARNING,
                    entity_type="product",
                    entity_id=product.id,
                    code="product_name_qualifier_not_source_literal",
                    message=(
                        f"Core product name {name_support['core']!r} is supported "
                        f"by cited evidence; parenthetical qualifier "
                        f"{name_support['qualifier']!r} was not textually "
                        f"supported as a whole and was treated as an optional "
                        f"display qualifier. Product retained unchanged "
                        f"(name={product.name!r})."
                    ),
                    evidence_ids=kept_ids,
                )
            )

        if product.product_group_id and product.product_group_id not in surviving_group_ids:
            issues.append(
                _issue(
                    severity=IssueSeverity.WARNING,
                    entity_type="product",
                    entity_id=product.id,
                    code="dangling_product_group_id",
                    message=(
                        f"product_group_id '{product.product_group_id}' is "
                        "dangling; cleared to None."
                    ),
                )
            )
            product.product_group_id = None

        surviving_products.append(product)

    # --- Requirement semantic source checks ---
    final_requirements: list[SubmittalRequirement] = []
    for req in surviving_requirements:
        evidence_items = _lookup_evidence(req.evidence_ids, by_id)
        if req.source_clause:
            clause_ok = any(
                _clause_supports(req.source_clause, item.source_clause)
                for item in evidence_items
            )
            if not clause_ok:
                issues.append(
                    _issue(
                        severity=IssueSeverity.WARNING,
                        entity_type="requirement",
                        entity_id=req.id,
                        code="unsupported_source_clause",
                        message=(
                            f"source_clause '{req.source_clause}' not supported "
                            "by cited evidence clauses."
                        ),
                        evidence_ids=req.evidence_ids,
                    )
                )
                # Keep requirement if cited evidence remains credible.
                if not evidence_items:
                    issues.append(
                        _issue(
                            severity=IssueSeverity.ERROR,
                            entity_type="requirement",
                            entity_id=req.id,
                            code="unsupported_source_clause_fatal",
                            message="Requirement rejected: unsupported clause and no evidence.",
                        )
                    )
                    rejected_entities.append(
                        {
                            "entity_type": "requirement",
                            "entity_id": req.id,
                            "reason": "unsupported_source_clause_fatal",
                        }
                    )
                    continue

        if req.source_category:
            if not _source_category_supported(req.source_category, evidence_items):
                issues.append(
                    _issue(
                        severity=IssueSeverity.WARNING,
                        entity_type="requirement",
                        entity_id=req.id,
                        code="unsupported_source_category",
                        message=(
                            f"source_category '{req.source_category}' not clearly "
                            "supported by cited evidence text/headings."
                        ),
                        evidence_ids=req.evidence_ids,
                    )
                )

        if req.condition:
            if not _condition_supported(req.condition, evidence_items):
                issues.append(
                    _issue(
                        severity=IssueSeverity.WARNING,
                        entity_type="requirement",
                        entity_id=req.id,
                        code="unsupported_condition",
                        message=(
                            f"condition '{req.condition}' not supported by cited "
                            "evidence; cleared to None."
                        ),
                        evidence_ids=req.evidence_ids,
                    )
                )
                stripped_conditions.append(
                    {
                        "requirement_id": req.id,
                        "condition": req.condition,
                    }
                )
                req.condition = None

        if req.cross_references:
            kept_refs: list[str] = []
            for xref in req.cross_references:
                if _cross_ref_supported(xref, evidence_items):
                    kept_refs.append(xref)
                else:
                    issues.append(
                        _issue(
                            severity=IssueSeverity.WARNING,
                            entity_type="requirement",
                            entity_id=req.id,
                            code="unsupported_cross_reference",
                            message=(
                                f"cross_reference '{xref}' not found in cited "
                                "evidence; stripped."
                            ),
                            evidence_ids=req.evidence_ids,
                        )
                    )
                    stripped_cross_references.append(
                        {"requirement_id": req.id, "cross_reference": xref}
                    )
            req.cross_references = kept_refs

        final_requirements.append(req)

    req_by_id = {req.id: req for req in final_requirements}
    prod_by_id = {product.id: product for product in surviving_products}

    # --- Links ---
    surviving_links: list[RequirementProductLink] = []
    for link in working.links:
        kept_ids = _filter_evidence_ids(
            entity_type="link",
            entity_id=link.id,
            evidence_ids=list(link.evidence_ids),
            allowed=allowed,
            issues=issues,
        )
        if not kept_ids:
            issues.append(
                _issue(
                    severity=IssueSeverity.ERROR,
                    entity_type="link",
                    entity_id=link.id,
                    code="no_valid_evidence",
                    message="Link rejected: no valid evidence IDs remain.",
                )
            )
            rejected_entities.append(
                {
                    "entity_type": "link",
                    "entity_id": link.id,
                    "reason": "no_valid_evidence",
                }
            )
            continue
        link.evidence_ids = kept_ids

        if link.requirement_id not in req_by_id:
            issues.append(
                _issue(
                    severity=IssueSeverity.ERROR,
                    entity_type="link",
                    entity_id=link.id,
                    code="dangling_requirement_id",
                    message=(
                        f"Link rejected: requirement_id '{link.requirement_id}' "
                        "does not exist in surviving requirements."
                    ),
                )
            )
            rejected_entities.append(
                {
                    "entity_type": "link",
                    "entity_id": link.id,
                    "reason": "dangling_requirement_id",
                    "requirement_id": link.requirement_id,
                }
            )
            continue

        if link.product_id not in prod_by_id:
            issues.append(
                _issue(
                    severity=IssueSeverity.ERROR,
                    entity_type="link",
                    entity_id=link.id,
                    code="dangling_product_id",
                    message=(
                        f"Link rejected: product_id '{link.product_id}' does not "
                        "exist in surviving products."
                    ),
                )
            )
            rejected_entities.append(
                {
                    "entity_type": "link",
                    "entity_id": link.id,
                    "reason": "dangling_product_id",
                    "product_id": link.product_id,
                }
            )
            continue

        requirement = req_by_id[link.requirement_id]
        product = prod_by_id[link.product_id]
        req_evidence = _lookup_evidence(requirement.evidence_ids, by_id)
        prod_evidence = _lookup_evidence(product.evidence_ids, by_id)
        link_evidence = _lookup_evidence(link.evidence_ids, by_id)

        phrase = link.source_phrase
        product_terms = [
            term
            for term in (phrase, product.name, product.normalized_name)
            if term
        ]

        def _any_term_in(items: Sequence[SourceEvidence]) -> bool:
            return any(_phrase_in_evidence(term, items) for term in product_terms)

        method = link.mapping_method
        if isinstance(method, str):
            method = MappingMethod(method)

        # source_phrase support against cited link evidence
        if phrase and not _phrase_in_evidence(phrase, link_evidence):
            # Also allow phrase on requirement evidence for explicit/normalized.
            if not _phrase_in_evidence(phrase, req_evidence):
                issues.append(
                    _issue(
                        severity=IssueSeverity.WARNING,
                        entity_type="link",
                        entity_id=link.id,
                        code="unsupported_source_phrase",
                        message=(
                            f"source_phrase '{phrase}' not found in cited link/"
                            "requirement evidence."
                        ),
                        evidence_ids=link.evidence_ids,
                    )
                )

        if method == MappingMethod.EXPLICIT:
            # Must be directly named in requirement-supporting evidence.
            if phrase:
                explicit_ok = _phrase_in_evidence(phrase, req_evidence)
            else:
                explicit_ok = _any_term_in(req_evidence)
            if not explicit_ok:
                # Try normalized criteria before inferred.
                req_has = _any_term_in(req_evidence) or (
                    phrase is not None and _phrase_in_evidence(phrase, req_evidence)
                )
                prod_has_fuller = _product_name_supported(product, prod_evidence)
                if req_has and prod_has_fuller and phrase and _phrase_in_evidence(
                    phrase, req_evidence
                ):
                    mapping_downgrades.append(
                        {
                            "link_id": link.id,
                            "from": MappingMethod.EXPLICIT.value,
                            "to": MappingMethod.NORMALIZED.value,
                            "reason": "explicit_not_supported_normalized_ok",
                        }
                    )
                    issues.append(
                        _issue(
                            severity=IssueSeverity.WARNING,
                            entity_type="link",
                            entity_id=link.id,
                            code="mapping_downgraded",
                            message="explicit unsupported; downgraded to normalized.",
                            evidence_ids=link.evidence_ids,
                        )
                    )
                    link.mapping_method = MappingMethod.NORMALIZED
                elif req_has or _any_term_in(link_evidence):
                    mapping_downgrades.append(
                        {
                            "link_id": link.id,
                            "from": MappingMethod.EXPLICIT.value,
                            "to": MappingMethod.INFERRED.value,
                            "reason": "explicit_not_supported",
                        }
                    )
                    issues.append(
                        _issue(
                            severity=IssueSeverity.WARNING,
                            entity_type="link",
                            entity_id=link.id,
                            code="mapping_downgraded",
                            message=(
                                "explicit unsupported; downgraded to inferred "
                                "with review_required=True."
                            ),
                            evidence_ids=link.evidence_ids,
                        )
                    )
                    link.mapping_method = MappingMethod.INFERRED
                    link.review_required = True
                    if link.confidence is None:
                        link.confidence = 0.5
                else:
                    issues.append(
                        _issue(
                            severity=IssueSeverity.ERROR,
                            entity_type="link",
                            entity_id=link.id,
                            code="unsupported_explicit_mapping",
                            message="Link rejected: explicit mapping not source-supported.",
                            evidence_ids=link.evidence_ids,
                        )
                    )
                    rejected_entities.append(
                        {
                            "entity_type": "link",
                            "entity_id": link.id,
                            "reason": "unsupported_explicit_mapping",
                            "requirement_id": link.requirement_id,
                            "product_id": link.product_id,
                        }
                    )
                    continue

        elif method == MappingMethod.NORMALIZED:
            req_side = False
            if phrase:
                req_side = _phrase_in_evidence(phrase, req_evidence)
            if not req_side:
                req_side = _any_term_in(req_evidence)
            prod_side = _product_name_supported(product, prod_evidence)
            if not (req_side and prod_side):
                if req_side or prod_side or _any_term_in(link_evidence):
                    mapping_downgrades.append(
                        {
                            "link_id": link.id,
                            "from": MappingMethod.NORMALIZED.value,
                            "to": MappingMethod.INFERRED.value,
                            "reason": "normalized_incomplete",
                        }
                    )
                    issues.append(
                        _issue(
                            severity=IssueSeverity.WARNING,
                            entity_type="link",
                            entity_id=link.id,
                            code="mapping_downgraded",
                            message=(
                                "normalized incomplete; downgraded to inferred "
                                "with review_required=True."
                            ),
                            evidence_ids=link.evidence_ids,
                        )
                    )
                    link.mapping_method = MappingMethod.INFERRED
                    link.review_required = True
                    if link.confidence is None:
                        link.confidence = 0.5
                else:
                    issues.append(
                        _issue(
                            severity=IssueSeverity.ERROR,
                            entity_type="link",
                            entity_id=link.id,
                            code="unsupported_normalized_mapping",
                            message="Link rejected: normalized mapping not source-supported.",
                            evidence_ids=link.evidence_ids,
                        )
                    )
                    rejected_entities.append(
                        {
                            "entity_type": "link",
                            "entity_id": link.id,
                            "reason": "unsupported_normalized_mapping",
                            "requirement_id": link.requirement_id,
                            "product_id": link.product_id,
                        }
                    )
                    continue

        # Re-read method after possible downgrade.
        method = link.mapping_method
        if isinstance(method, str):
            method = MappingMethod(method)

        if method == MappingMethod.INFERRED:
            if not link.review_required:
                link.review_required = True
                issues.append(
                    _issue(
                        severity=IssueSeverity.WARNING,
                        entity_type="link",
                        entity_id=link.id,
                        code="inferred_missing_review_required",
                        message="inferred link missing review_required; set True.",
                        evidence_ids=link.evidence_ids,
                    )
                )
            if link.confidence is None:
                issues.append(
                    _issue(
                        severity=IssueSeverity.ERROR,
                        entity_type="link",
                        entity_id=link.id,
                        code="inferred_missing_confidence",
                        message="Link rejected: inferred mapping requires confidence.",
                        evidence_ids=link.evidence_ids,
                    )
                )
                rejected_entities.append(
                    {
                        "entity_type": "link",
                        "entity_id": link.id,
                        "reason": "inferred_missing_confidence",
                    }
                )
                continue
            plausible = (
                _any_term_in(req_evidence)
                or _any_term_in(prod_evidence)
                or _any_term_in(link_evidence)
            )
            if not plausible:
                issues.append(
                    _issue(
                        severity=IssueSeverity.ERROR,
                        entity_type="link",
                        entity_id=link.id,
                        code="unsupported_inferred_mapping",
                        message="Link rejected: inferred mapping has no plausible evidence.",
                        evidence_ids=link.evidence_ids,
                    )
                )
                rejected_entities.append(
                    {
                        "entity_type": "link",
                        "entity_id": link.id,
                        "reason": "unsupported_inferred_mapping",
                        "requirement_id": link.requirement_id,
                        "product_id": link.product_id,
                    }
                )
                continue

        surviving_links.append(link)

    cleaned = SubmittalExtractionResult(
        requirements=final_requirements,
        product_groups=surviving_groups,
        products=surviving_products,
        links=surviving_links,
    )

    warning_codes = Counter(
        issue.code for issue in issues if issue.severity == IssueSeverity.WARNING
    )
    error_codes = Counter(
        issue.code for issue in issues if issue.severity == IssueSeverity.ERROR
    )

    report = ValidationReport(
        valid=not any(issue.severity == IssueSeverity.ERROR for issue in issues),
        issues=issues,
        accepted_counts={
            "requirements": len(cleaned.requirements),
            "product_groups": len(cleaned.product_groups),
            "products": len(cleaned.products),
            "links": len(cleaned.links),
        },
        rejected_counts={
            "requirements": sum(
                1 for item in rejected_entities if item["entity_type"] == "requirement"
            ),
            "product_groups": sum(
                1
                for item in rejected_entities
                if item["entity_type"] == "product_group"
            ),
            "products": sum(
                1 for item in rejected_entities if item["entity_type"] == "product"
            ),
            "links": sum(
                1 for item in rejected_entities if item["entity_type"] == "link"
            ),
        },
        warning_counts=dict(warning_codes),
        mapping_downgrades=mapping_downgrades,
        stripped_cross_references=stripped_cross_references,
        stripped_conditions=stripped_conditions,
        rejected_entities=rejected_entities,
    )
    # Attach error code tallies into warning_counts-style map for visibility.
    report.warning_counts = {
        **{f"warning:{k}": v for k, v in warning_codes.items()},
        **{f"error:{k}": v for k, v in error_codes.items()},
    }

    return ValidatedExtraction(extraction=cleaned, report=report)


__all__ = [
    "IssueSeverity",
    "ValidatedExtraction",
    "ValidationIssue",
    "ValidationReport",
    "validate_extraction",
]

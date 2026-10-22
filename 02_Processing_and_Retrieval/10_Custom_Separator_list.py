from langchain_text_splitters import RecursiveCharacterTextSplitter

LEGAL_SEPARATORS = [
    r"\nARTICLE\s+[IVXLC0-9]+\b",       # ARTICLE I, ARTICLE II, ARTICLE 1
    r"\nSECTION\s+\d+(?:\.\d+)*\b",  # SECTION 1, SECTION 2.1
    r"\nSection\s+\d+(?:\.\d+)*\b",  # Section 3.2
    r"\n\d+(?:\.\d+)+[.)]?\s+",      # 1.1, 2.3.1, 4.2(a)
    r"\n\d+[.)]\s+",                 # 1. Definitions, 2) Obligations
    r"\n\([a-z]\)\s+",               # (a), (b), (c)
    r"\n\([ivxlcdm]+\)\s+",          # (i), (ii), (iii)
    r"\n\n",                         # Paragraph breaks
    r"\n",                           # Line breaks
    r"(?<=[.!?])\s+",                # Sentence boundaries
    r"\s+",                          # Whitespace
    "",
]

ACME_MEMO_EXCERPT = """
Executive Summary

Acme Corporation enters Q3 2024 with strong momentum across our product portfolio.
Product Alpha achieved Q2 revenue of $12.4M, representing 34% year-over-year growth
and bringing our ARR to $48.2M. Enterprise adoption continues to accelerate, with
23 new logos added in Q2 alone.

Product Beta stabilized at $3.8M Q2 revenue after the pricing restructure in Q1.
While churn remains elevated at 4.2% (target: <3%), the new onboarding flow has
reduced time-to-value by 40%. We expect churn improvements to materialize in Q3
as the cohort matures.

Product Gamma remains in private beta with 47 design partners. Launch is scheduled
for September 15, 2024. Early feedback indicates strong product-market fit in the
mid-market segment, with a projected ASP of $45K.

Q3 Financial Targets
Combined Q3 revenue target: $18.6M
- Product Alpha: $14.2M (Q3 target)
- Product Beta: $4.0M (Q3 target)
- Product Gamma: $0.4M (post-launch partial quarter)

Risk Register
1. Competitive threat from Nexus Corp's new enterprise offering (Impact: High,
Probability: Medium). Mitigation: Accelerate Alpha's API platform release.
2. Beta churn exceeding 5% threshold (Impact: Medium, Probability: Medium).
Mitigation: Dedicated customer success team for at-risk accounts.
3. Gamma launch delay beyond September 30 (Impact: High, Probability: Low).
Mitigation: Weekly launch readiness reviews with engineering leads.
4. Key person dependency on Alpha's ML infrastructure team (Impact: Medium,
Probability: Low). Mitigation: Cross-training program initiated in Q2.
5. Regulatory changes in EU market affecting data processing capabilities
(Impact: High, Probability: Low). Mitigation: Legal team monitoring;
architecture review for data residency options.
"""

splitter = RecursiveCharacterTextSplitter(
    separators=LEGAL_SEPARATORS,
    is_separator_regex=True,
    chunk_size=800,
    chunk_overlap=100,
    keep_separator="start",
)

chunks = splitter.split_text(ACME_MEMO_EXCERPT)

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
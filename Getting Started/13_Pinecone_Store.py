"""
Demonstrates:
- Creating an index
- Upserting vectors with metadata
- Querying with similarity search
- Filtering by metadata
- Updating and deleting vectors
"""
import os
import time
from ollama import Client
from pinecone import Pinecone, ServerlessSpec

# Acme memo chunks with IDs and metadata
ACME_CHUNKS = [
    {
        "id": "acme-q3-exec-summary",
        "text": "Acme Corporation enters Q3 2025 with strong momentum from Product Alpha's " 
"enterprise launch but faces headwinds from increased competition in the "
"mid-market segment. This memo outlines our strategic priorities, resource " 
"allocation, and risk mitigation plans for the quarter.", 
        "section": "Executive Summary",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-alpha",
        "text": "Product Alpha (Enterprise Analytics Platform) "
"Q2 revenue: $12.4M (18% above forecast). Alpha secured 23 new enterprise "
"contracts including Fortune 500 wins at Meridian Healthcare ($1.2M ACV) and "
"Titan Manufacturing ($890K ACV). Current ARR run rate: $48.2M. The v3.2 "
"release added real-time dashboard streaming, which was the #1 feature "
"request from enterprise buyers. Engineering headcount: 34 FTEs. "
"Key risk: Dataview Inc. launched a competing product at 40% lower price "
"point. Three pipeline deals ($2.1M combined) are now in competitive "
"review. Sales team reports Dataview is winning on price but losing on "
"reliability and integration depth.",
        "section": "Product Alpha",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-beta",
        "text": "Product Beta (SMB Reporting Tool) "
"Q2 revenue: $3.8M (6% below forecast). Churn increased to 4.2% monthly, "
"up from 2.8% in Q1. Root cause analysis points to poor onboarding "
"experience — 62% of churned customers never completed initial setup. "
"Customer satisfaction (CSAT) dropped from 4.1 to 3.6. We are launching "
""Beta Quick Start" program in Q3 with dedicated onboarding specialists. "
"Target: reduce churn to 2.5% by end of Q3. "
"Budget allocation: $1.8M for onboarding improvement, $600K for UX "
"redesign of setup wizard.",
        "section": "Product Beta",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-gamma",
        "text": "Product Gamma (AI Data Assistant — New) "
"Status: Private beta with 12 design partners. Launch planned for September "
"15, 2025. Early feedback is positive — 9 of 12 partners rated the "
"natural language query feature as "transformative." Pricing strategy under "
"review: considering $45/user/month for teams, $120/user/month for "
"enterprise with advanced governance features. "
"Competitive landscape: Four well-funded startups in this space. Our "
"advantage is deep integration with Alpha's data layer. Estimated TAM: $2.8B by 2027. "
"Go-to-market plan requires $3.2M Q3 investment across product marketing "
"($1.4M), developer relations ($800K), and launch events ($1M).",
        "section": "Product Gamma",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-financial",
        "text": "Q3 revenue target: $18.6M (combined). Operating margin target: 12%. "
"Headcount plan: 14 new hires across engineering (8), sales (4), and "
"customer success (2). Total compensation budget increase: $2.4M annualized.",
        "section": "Financial Outlook",
        "source": "Q3 Strategy Memo",
    },
    {
        "id": "acme-q3-risks",
        "text": "Risk Register: 1. Dataview pricing pressure on Alpha (Impact: High, Likelihood: Medium) "
"2. Beta churn trajectory if onboarding fix underperforms (Impact: Medium, Likelihood: Medium) "
"3. Gamma launch delay if API stability issues persist (Impact: High, Likelihood: Low) "
"4. Macroeconomic slowdown affecting enterprise sales cycles (Impact: High, Likelihood: Medium) "
"5. Key person risk — 3 senior engineers on Alpha have received external offers (Impact: High, Likelihood: Medium)",
        "section": "Risk Register",
        "source": "Q3 Strategy Memo",
    },
]

# ============================================================
# CONFIGURATION
# ============================================================
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "acme-rag-demo"
EMBEDDING_MODEL = "nomic-embed-text:latest"
EMBEDDING_DIMENSIONS = 768
CLOUD = "aws"
REGION = "us-east-1"
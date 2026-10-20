"""
Extract text from messy PDFs with tables,
headers/footers, and multi-column layouts.
"""
import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import pdfplumber
import pymupdf as fitz
from PIL import Image
import pytesseract
import io


@dataclass
class ExtractedTable:
    """A table extracted from a PDF page."""
    page_number: int
    rows: List[List[str]]
    headers: List[str]
    bbox: Optional[Tuple[float, float, float, float]] = None

    def to_markdown(self) -> str:
        """COnvert table to markdown format for chunking."""
        if not self.rows:
            return ""
        lines = []
        # header row
        lines.append("| " + " | ".join(self.headers)+" |")
        lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
        # Data rows
        for row in self.rows:
            # Pad row if it has fewer columns than headers
            padded = row + [""] * (len(self.headers) - len(row))
            lines.append("| " + " | ".join(padded[:len(self.headers)]) + " |")
        return "\n".join(lines)

    def to_json(self) -> List[Dict[str, str]]:
        """Convert table to list of dicts for structured access"""
        return [
            {h: row[i] if i < len(row) else "" for i,
             h in enumerate(self.headers)}
            for row in self.rows
        ]


@dataclass
class PageContent:
    """Extracted content from a single PDF page"""
    page_number: int
    text: str
    tables: List[ExtractedTable] = field(default_factory=list)
    is_scanned: bool = False


def detect_headers_footers(pages_text: List[str], threshold: int = 3) -> Dict[str, List[str]]:
    """
    Detect repeated text across pages that likely represents headers/footers.
    Strategy: Text that appears identically on `threshold` or more pages,
    positioned at the top or bottom of the page, is likely a header/footer.
    Args:
        pages_text: List of full page text, one string per page
        threshold: Minimum number of pages a line must appear on
    Returns:
        Dict with 'headers' and 'footers' lists
    """

    # Extract first and last 3 lines from each page
    top_lines = Counter()
    bottom_lines = Counter()
    for page_text in pages_text:
        lines = [line.strip()
                 for line in page_text.split('\n') if line.strip()]
        if not lines:
            continue

        # Top 3 lines
        for line in lines[:3]:
            top_lines[line] += 1

        # Bottom 3 lines
        for line in lines[-3:]:
            bottom_lines[line] += 1

    # Lines appearing on threshold pages are headers/footers
    headers = [line for line, count in top_lines.items() if count >= threshold]
    footers = [line for line, count in bottom_lines.items() if count >=
               threshold]

    # Also catch page number patterns
    page_num_pattern = re.compile(
        r'^(Page\s+\d+(\s+of\s+\d+)?|\d+\s*$|—\s*\d+\s*—|\[\d+\])',
        re.IGNORECASE
    )
    for line_counter in [top_lines, bottom_lines]:
        for line in line_counter:
            if page_num_pattern.match(line) and line not in headers+footers:
                footers.append(line)

    return {"headers": headers, "footers": footers}


def remove_headers_footers(text: str, headers_footers: Dict[str, List[str]]) -> str:
    """remove detected headers and footers from page text"""
    lines = text.split('\n')
    all_noise = set(headers_footers["headers"] + headers_footers["footers"])
    cleaned = [line for line in lines if line.strip() not in all_noise]
    # Also remove page number patterns inline
    page_num_patterns = [
        r'Page\s+\d+\s+of\s+\d+',
        r'^\s*\d+\s*$',
        r'—\s*\d+\s*—',
    ]
    result = '\n'.join(cleaned)
    for pattern in page_num_patterns:
        result = re.sub(pattern, '', result, flags=re.MULTILINE)
    return result.strip()


def extract_tables_pdfplumber(pdf_path: str) -> List[ExtractedTable]:
    """
    Extract tables from a PDF using pdfplumber.
    pdfplumber detects tables by finding intersecting lines that form
    a grid pattern. It handles:
    - Bordered tables
    - Tables with partial borders
    - Tables spanning most of the page width
    It struggles with:
    - Borderless tables (no lines at all)
    - Tables made with tabs/spaces only
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 5,
                    "join_tolerance": 5,
                    "min_words_vertical": 2,
                    "min_words_horizontal": 2,
                }
            )
            for table_data in page_tables:
                if not table_data or len(table_data) < 2:
                    continue
                # First row is typically headers
                headers = [str(cell or "").strip() for cell in table_data[0]]
                rows = [
                    [str(cell or "").strip() for cell in row]
                    for row in table_data[1:]
                ]
                tables.append(ExtractedTable(
                    page_number=page_num,
                    rows=rows,
                    headers=headers,
                ))

        return tables


def extract_with_column_detection(pdf_path: str) -> List[PageContent]:
    """
    Extract text from PDF with multi-column layout detection.
    Uses pymupdf (fitz) to get text blocks with bounding boxes,
    then sorts by column position before reading order.
    """
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, 1):
            # Get text blocks with position info
            blocks = page.get_text("dict")["blocks"]
            # Separate text blocks from image blocks
            text_blocks = []
            has_images = False
            for block in blocks:
                if block["type"] == 0:  # text block
                    text_blocks.append(block)
                elif block["type"] == 1:  # image block
                    has_images = True

            if not text_blocks:
                pages.append(PageContent(
                    page_number=page_num,
                    text="",
                    is_scanned=has_images,
                ))
                continue

            # Detect columns by analysing x-coordinate clusters
            x_positions = [block["bbox"][0] for block in text_blocks]
            columns = detect_columns(x_positions, page.rect.width)
            if len(columns) > 1:
                # Multi-column: sort blocks by column, then y-position
                sorted_blocks = sort_by_columns(text_blocks, columns)
            else:
                # Sinlge column: sort by y-position (top to bottom)
                sorted_blocks = sorted(text_blocks, key=lambda b: b["bbox"][1])

            # Extract text from sorted blocks
            page_text = ""
            for block in sorted_blocks:
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    if line_text.strip():
                        page_text += line_text.strip() + "\n"
                page_text += "\n"

            pages.append(PageContent(
                page_number=page_num,
                text=page_text.strip(),
                is_scanned=has_images and not page_text.strip(),
            ))
    return pages


def detect_columns(x_positions: List[float], page_width: float) -> List[Tuple[float, float]]:
    """
    Detect column boundaries from text block x-coordinates.
    Returns list of (left_edge, right_edge) tuples for each column.
    """
    if not x_positions:
        return [(0, page_width)]
    # Cluster x-postions using a simple gap-based method
    sorted_x = sorted(set(x_positions))
    if len(sorted_x) < 2:
        return [(0, page_width)]
    # Find gaps larger than 15% of page width
    min_gap = page_width * 0.15
    columns = []
    col_start = sorted_x[0]
    for i in range(1, len(sorted_x)):
        gap = sorted_x[i] - sorted_x[i-1]
        if gap > min_gap:
            columns.append((col_start, sorted_x[i-1] + min_gap / 2))
            col_start = sorted_x[i]

    columns.append((col_start, page_width))
    return columns


def sort_by_columns(blocks: List[Dict], columns: List[Tuple[float, float]]) -> List[Dict]:
    """
    Sort text blocks: left column top-to-bottom, then right column top-to-bottom
    """
    column_blocks = {i: [] for i in range(len(columns))}
    for block in blocks:
        block_x = block["bbox"][0]
        # Assign block to nearest column
        best_col = 0
        best_dist = float("inf")
        for i, (left, right) in enumerate(columns):
            dist = abs(block_x - left)
            if dist < best_dist:
                best_dist = dist
                best_col = i
        column_blocks[best_col].append(block)
    # Sort each column by y-postion, then concatenate columsn
    sorted_blocks = []
    for col_idx in sorted(column_blocks.keys()):
        col = sorted(column_blocks[col_idx], key=lambda b: b["bbox"][1])
        sorted_blocks.extend(col)
    return sorted_blocks


def ocr_page(pdf_path: str, page_num: int, dpi: int = 300) -> str:
    """
    OCR a single page of a scanned PDF.
    Uses pymupdf to render the page as an image, then pytesseract for OCR.
    Args:
        pdf_path: Path to the PDF file
        page_num: 0-indexed page number
        dpi: Resolution for rendering (300 recommended minimum)
    """
    with fitz.open(pdf_path) as doc:
        page = doc[page_num]
        # Render page as image
        mat = fitz.Matrix(dpi / 72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        # Convert to PIL Image
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        # Preprocess for better OCR
        img = img.convert("L")  # Grayscale
        # OCR with pytesseract
        text = pytesseract.image_to_string(
            img,
            config="--psm 6 --oem 3",  # Assume uniform block of text, LSTM engine
        )

    return text


def extract_section_hierarchy(text: str) -> List[Dict]:
    """
    Detect section headers and build a hierarchy.
    Looks for patterns like:
    - "1. Section Title"
    - "1.1 Subsection Title"
    - "## Markdown Headers"
    - ALL CAPS LINES (common in legal/financial docs)
    """
    lines = text.split('\n')
    sections = []
    current_path = []
    # patterns for section headers
    numbered_pattern = re.compile(r'^(\d+(?:\.\d+)*)[.)\s]+(.+)$')
    caps_pattern = re.compile(r'^[A-Z][A-Z\s]{5,}$')
    markdown_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # Check numbered sections (1., 1.1, 2.3.1)
        match = numbered_pattern.match(line)
        if match:
            number = match.group(1)
            title = match.group(2).strip()
            level = number.count('.') + 1
            sections.append({
                "line_index": i,
                "level": level,
                "number": number,
                "title": title,
                "full_header": line,
            })
            continue

        # check all caps headers
        if caps_pattern.match(line) and len(line.split()) <= 8:
            sections.append({
                "line_index": i,
                "level": 1,
                "number": "",
                "title": line.title(),
                "full_header": line,
            })
            continue

        # Check markdown headers
        match = markdown_pattern.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            sections.append({
                "line_index": i,
                "level": level,
                "number": "",
                "title": title,
                "full_header": line,
            })

    return sections


def chunk_with_structure(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Chunk text while preserving document structure.
    Each chunk includes:
    - The text content
    - The section hierarchy (breadcrumb) it belongs to
    - Page number if available
    """
    sections = extract_section_hierarchy(text)
    lines = text.split('\n')
    chunks = []
    current_chunk_lines = []
    current_section_path = []
    current_size = 0
    for i, line in enumerate(lines):
        # Check if this line is a section header
        for section in sections:
            if section["line_index"] == i:
                # update sectiuon path
                level = section["level"]
                # Trim path to current level and add new section
                current_section_path = [
                    s for s in current_section_path if s["level"] < level
                ]
                current_section_path.append(section)
                break
        current_chunk_lines.append(line)
        current_size += len(line) + 1  # +1 for new line
        if current_size >= max_chunk_size:
            chunk_text = '\n'.join(current_chunk_lines).strip()
            if chunk_text:
                breadcrumb = " > ".join(s["title"]
                                        for s in current_section_path)
                chunks.append({
                    "text": chunk_text,
                    "section": breadcrumb or "Unknown Section",
                    "metadata": {
                        "section_path": [s["title"] for s in current_section_path],
                        "section_level": current_section_path[-1]["level"]
                        if current_section_path else 0,
                        "char_count": len(chunk_text),
                    },
                })
            # Keep overlap
            overlap_lines = []
            overlap_size = 0
            for line in reversed(current_chunk_lines):
                if overlap_size + len(line) > overlap:
                    break
                overlap_lines.insert(0, line)
                overlap_size += len(line) + 1

            current_chunk_lines = overlap_lines
            current_size = overlap_size
    # Dont forget the last chunk
    if current_chunk_lines:
        chunk_text = '\n'.join(current_chunk_lines).strip()
        if chunk_text:
            breadcrumb = ' > '.join(s["title"] for s in current_section_path)
            chunks.append({
                "text": chunk_text,
                "section": breadcrumb or "Unknown Section",
                "metadata": {
                    "section_path": [s["title"] for s in current_section_path],
                    "section_level": current_section_path[-1]["level"]
                    if current_section_path else 0,
                    "char_count": len(chunk_text),
                },
            })
    return chunks


class PDFExtractionPipeline:
    """
    Production PDF extraction pipeline.
    Handles:
        1. Header/footer detection and removal
        2. Table extraction and markdown conversion
        3. Multi-column layout detection
        4. OCR for scanned pages
        5. Section hierarchy preservation
        6. Structure-aware chunking
    """
    def __init__(self,
                 pdf_path: str, chunk_size: int = 500,
                 chunk_overlap: int = 50, ocr_dpi: int = 300):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_dpi = ocr_dpi

    def extract(self) -> List[Dict]:
        """
        Full extraction pipeline.
        Returns list of chunks with text and metadata.
        """
        print(f"[1/5] Extracting pages with column detection...")
        pages = extract_with_column_detection(self.pdf_path)

        print(f"[2/5] Detecting headers and footers...")
        pages_text = [p.text for p in pages]
        hf = detect_headers_footers(pages_text)
        print(f" Found {len(hf['headers'])} header patterns, "
              f"{len(hf['footers'])} footer patterns")

        print(f"[3/5] Extracting tables...")
        tables = extract_tables_pdfplumber(self.pdf_path)
        print(f" Found {len(tables)} tables")

        print(f"[4/5] Processing scanned pages with OCR...")
        for page in pages:
            if page.is_scanned:
                print(f" OCR on page {page.page_number}...")
                page.text = ocr_page(
                    self.pdf_path, page.page_number - 1, self.ocr_dpi
                )

        print(f"[5/5] Chunking with structure preservation...")
        # Combine all text, clean headers/footers
        full_text = ""
        for page in pages:
            cleaned = remove_headers_footers(page.text, hf)
            full_text += f"\n\n[Page {page.page_number}]\n{cleaned}"

        # Insert table markdown at appropriate positions
        for table in tables:
            table_md = f"\n\n[Table from page {table.page_number}]\n{table.to_markdown()}\n"
            # Append tables after the page marker
            marker = f"[Page {table.page_number}]"
            full_text = full_text.replace(
                marker, marker + table_md, 1
            )

        # Chunk with structure
        chunks = chunk_with_structure(
            full_text,
            max_chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        print(f" Generated {len(chunks)} chunks")
        return chunks

#Sample text: excerpt from the Acme Corp Q3 Strategy Memo
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


# Demo usage ##############
def demo_with_sample_text():
    """
    Demo the extraction pipeline using the Acme memo text.
    For a real PDF demo, use: PDFExtractionPipeline("path/to/file.pdf").extract()
    """
    print("=" * 70)
    print(" Messy PDF Extraction Demo")
    print(" (Using Acme Corp memo text as sample)")
    print("=" * 70)
    
    # Simulate what the pipeline produces
    chunks = chunk_with_structure(ACME_MEMO_EXCERPT, max_chunk_size=400, overlap=50)
    print(f"\nGenerated {len(chunks)} structure-aware chunks:\n")

    for i, chunk in enumerate(chunks):
        print(f" Chunk {i+1}:")
        print(f" Section: {chunk['section']}")
        print(f" Size: {chunk['metadata']['char_count']} chars")
        preview = chunk['text'][:120].replace('\n', ' ')
        print(f" Preview: {preview}...")
        print()

    # Demo table extraction to markdown
    print("=" * 70)
    print(" Table Extraction Demo")
    print("=" * 70)

    sample_table = ExtractedTable(
        page_number=2,
        headers=["Product", "Q2 Revenue", "ARR", "Churn"],
        rows=[
            ["Alpha", "$12.4M", "$48.2M", "1.8%"],
            ["Beta", "$3.8M", "$14.1M", "4.2%"],
            ["Gamma", "(beta)", "—", "—"],
        ],
    )

    print(f"\n Extracted table from page {sample_table.page_number}:")
    print(f"\n As Markdown (for chunking):")
    for line in sample_table.to_markdown().split('\n'):
        print(f" {line}")

    print(f"\n As JSON (for structured access):")
    for row in sample_table.to_json():
        print(f" {json.dumps(row)}")
    
    # Demo header/footer detection
    print(f"\n{'='*70}")
    print(" Header/Footer Detection Demo")
    print(f"{'='*70}")
    sample_pages = [
        "Acme Corporation — Confidential\nQ3 Strategy Memo\n\nContent page 1...\n\nPage 1 of 5",
        "Acme Corporation — Confidential\nQ3 Strategy Memo\n\nContent page 2...\n\nPage 2 of 5",
        "Acme Corporation — Confidential\nQ3 Strategy Memo\n\nContent page 3...\n\nPage 3 of 5",
        "Acme Corporation — Confidential\nQ3 Strategy Memo\n\nContent page 4...\n\nPage 4 of 5",
    ]

    hf = detect_headers_footers(sample_pages, threshold=3)
    print(f"\n Detected headers: {hf['headers']}")
    print(f" Detected footers: {hf['footers']}")

    cleaned = remove_headers_footers(sample_pages[0], hf)
    print(f"\n Before cleaning:\n {sample_pages[0]}")
    print(f"\n After cleaning:\n {cleaned}")

if __name__ == "__main__":
    demo_with_sample_text()
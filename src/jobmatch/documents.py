"""Document loading. CONTRACT (implementation owned by the graph task, used by the API):

async def load_document(data: bytes, filename: str) -> str

Converts an uploaded offer/CV to plain text:
  - .txt / .md            -> decoded UTF-8 text
  - .pdf with a text layer -> text extracted with pypdf
  - .pdf with too little text (< settings.min_text_chars_for_pdf, i.e. scanned)
                           -> pages rendered with pypdfium2, transcribed by the vision LLM
  - .png / .jpg / .jpeg / .webp -> transcribed by the vision LLM (`jobmatch.llm.image_message`)
Raises UnsupportedDocumentError for any other type or an empty result.
"""

import asyncio
import io
from pathlib import Path

from jobmatch import llm
from jobmatch.config import get_settings

TEXT_SUFFIXES = {".txt", ".md"}
IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
SCAN_DPI = 150
MAX_SCANNED_PAGES = 3

TRANSCRIBE_PROMPT = (
    "Transcribe all the text of this document image (a job offer or a CV) faithfully, in its "
    "original language. Keep the reading order and the structure (headings, bullet lists) "
    "using plain text. Do not summarize, translate, comment or add anything. "
    "Output only the transcription."
)


class UnsupportedDocumentError(ValueError):
    """The uploaded file type is not supported or contains no readable text."""


async def load_document(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = _decode_text(data)
    elif suffix == ".pdf":
        text = await _load_pdf(data)
    elif suffix in IMAGE_MIME:
        text = await _transcribe_image(data, IMAGE_MIME[suffix])
    else:
        raise UnsupportedDocumentError(f"Unsupported file type: {suffix or filename!r}")
    text = text.strip()
    if not text:
        raise UnsupportedDocumentError(f"No readable text found in {filename!r}")
    return text


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise UnsupportedDocumentError("Text files must be UTF-8 encoded") from e


async def _load_pdf(data: bytes) -> str:
    text = await asyncio.to_thread(_extract_pdf_text, data)
    if len(text.strip()) >= get_settings().min_text_chars_for_pdf:
        return text
    # Little or no text layer: probably a scanned document -> OCR with the vision model.
    images = await asyncio.to_thread(_render_pdf_pages, data)
    pages = await asyncio.gather(*(_transcribe_image(img, "image/png") for img in images))
    return "\n\n".join(p.strip() for p in pages)


def _extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except (PyPdfError, ValueError, OSError) as e:
        raise UnsupportedDocumentError(f"Unreadable PDF: {e}") from e


def _render_pdf_pages(data: bytes) -> list[bytes]:
    """Render the first pages to PNG (pypdfium2 renders at 72 dpi for scale=1)."""
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        raise UnsupportedDocumentError(f"Unreadable PDF: {e}") from e
    try:
        images = []
        for i in range(min(len(pdf), MAX_SCANNED_PAGES)):
            pil = pdf[i].render(scale=SCAN_DPI / 72).to_pil()
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            images.append(buf.getvalue())
        return images
    finally:
        pdf.close()


async def _transcribe_image(image: bytes, mime: str) -> str:
    reply = await llm.get_llm(temperature=0.0).ainvoke(
        [llm.image_message(TRANSCRIBE_PROMPT, image, mime)]
    )
    content = reply.content
    if isinstance(content, list):  # content parts
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)

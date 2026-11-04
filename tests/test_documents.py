"""Tests of `load_document` (the vision model is faked)."""

import io

import pypdfium2 as pdfium
import pytest
from PIL import Image

from jobmatch.documents import MAX_SCANNED_PAGES, UnsupportedDocumentError, load_document

LOREM = "Data Engineer confirme - Python, SQL, Apache Spark, Airflow, Docker. " * 5


def make_text_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF with a real text layer (Helvetica, ASCII text)."""
    stream = f"BT /F1 10 Tf 40 800 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = io.BytesIO(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % i + obj + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    out.write(b"".join(b"%010d 00000 n \n" % o for o in offsets))
    out.write(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    )
    return out.getvalue()


def make_blank_pdf(pages: int) -> bytes:
    """A PDF without any text layer, like a scanned document."""
    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(595, 842)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def make_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buf, format="PNG")
    return buf.getvalue()


def _image_mime(messages) -> str:
    [msg] = messages
    return msg.content[1]["image_url"]["url"].split(";")[0].removeprefix("data:")


async def test_txt_and_md():
    assert await load_document("Bonjour, été\n".encode(), "cv.txt") == "Bonjour, été"
    assert await load_document(b"\xef\xbb\xbf# Titre", "offer.MD") == "# Titre"


async def test_text_pdf_uses_text_layer(install_fake_llm):
    fake = install_fake_llm()
    text = await load_document(make_text_pdf(LOREM), "cv.pdf")
    assert "Apache Spark" in text
    assert not fake.calls  # no vision call


async def test_scanned_pdf_uses_vision_on_first_pages(install_fake_llm):
    fake = install_fake_llm(text_response="Texte transcrit")
    text = await load_document(make_blank_pdf(MAX_SCANNED_PAGES + 2), "scan.pdf")
    assert text == "\n\n".join(["Texte transcrit"] * MAX_SCANNED_PAGES)
    assert len(fake.calls["text"]) == MAX_SCANNED_PAGES
    assert _image_mime(fake.calls["text"][0]) == "image/png"


@pytest.mark.parametrize(("name", "mime"), [("cv.png", "image/png"), ("cv.JPG", "image/jpeg")])
async def test_image_uses_vision(install_fake_llm, name, mime):
    fake = install_fake_llm(text_response="  CV de Camille  ")
    assert await load_document(make_png(), name) == "CV de Camille"
    assert _image_mime(fake.calls["text"][0]) == mime


@pytest.mark.parametrize(
    ("data", "name"),
    [
        (b"PK\x03\x04", "cv.docx"),
        (b"data", "noextension"),
        (b"   \n", "empty.txt"),
        (b"\xff\xfe\x00bad", "latin.txt"),
        (b"not a pdf", "broken.pdf"),
    ],
)
async def test_unsupported_or_empty(install_fake_llm, data, name):
    install_fake_llm(text_response="")
    with pytest.raises(UnsupportedDocumentError):
        await load_document(data, name)


async def test_empty_vision_result_is_rejected(install_fake_llm):
    install_fake_llm(text_response="")
    with pytest.raises(UnsupportedDocumentError, match="No readable text"):
        await load_document(make_png(), "blank.png")

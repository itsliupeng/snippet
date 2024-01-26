import fitz
from PIL import Image
import cv2
import numpy as np
import pypdfium2
import io
import pdf2image
import pypdf

pdf_path = "/ML-A100/home/renxiaoyi/data/books/scimag/ocr/part_00/part_0/jabr.2000.8746.pdf"
scale = 96 / 72


def t1():
    imgs = []
    with fitz.open(pdf_path) as pdf:
        for pg in range(0, pdf.page_count):
            page = pdf[pg]
            mat = fitz.Matrix(scale, scale)
            pm = page.get_pixmap(matrix=mat, alpha=False)

            # if width or height > 2000 pixels, don't enlarge the image
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            imgs.append(img)
    return imgs


def t2():
    pdf = pypdfium2.PdfDocument(pdf_path)
    pages = range(len(pdf))
    pils = []
    for i in pages:
        image = pdf[i].render(scale=scale).to_pil()
        pils.append(image)
    return pils


def t3():
    pils = []
    pdf = pypdf.PdfReader(pdf_path)
    pages = range(len(pdf.pages))
    for i in pages:
        page_bytes = io.BytesIO()
        writer = pypdf.PdfWriter()
        writer.add_page(pdf.pages[i])
        writer.write(page_bytes)
        page_bytes = page_bytes.getvalue()
        img = pdf2image.convert_from_bytes(
            page_bytes,
            dpi=96,
            fmt="png",
            output_folder=None,
            single_file=True,
            output_file="%02d" % (i + 1),
        )[0]
        return_pil = True
        if return_pil:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=img.format)
            pils.append(img_bytes)
    return pils
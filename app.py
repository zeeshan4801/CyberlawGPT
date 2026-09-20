# ==========================================
# DOWNLOAD PDF FROM GOOGLE DRIVE
# ==========================================

import gdown


PDF_FILE = "1470910659_707.pdf"


GOOGLE_DRIVE_FILE_ID = "PUT_YOUR_FILE_ID_HERE"



def download_pdf():

    if not os.path.exists(PDF_FILE):

        url = (
            f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        )

        gdown.download(
            url,
            PDF_FILE,
            quiet=False
        )


download_pdf()

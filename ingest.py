"""
ingest.py
---------
CLI batch ingestion: processes every PDF in /data at once.
Run this once (and again any time you add new PDFs directly to /data
instead of using the in-app uploader):
    python ingest.py
"""

import os
import glob
from vectorstore_utils import ingest_files, get_indexed_sources, get_vectorstore, DATA_DIR


def main():
    doc_paths = []
    for ext in ["*.pdf", "*.txt", "*.md", "*.csv"]:
        doc_paths.extend(glob.glob(os.path.join(DATA_DIR, ext)))

    if not doc_paths:
        raise FileNotFoundError(
            f"No documents found in ./{DATA_DIR}/. Drop your files (.pdf, .txt, .md, .csv) there before running ingest.py"
        )

    vectorstore = get_vectorstore()
    already_indexed = get_indexed_sources(vectorstore)

    new_paths = [p for p in doc_paths if os.path.basename(p) not in already_indexed]
    skipped = len(doc_paths) - len(new_paths)
    if skipped:
        print(f"Skipping {skipped} file(s) already indexed.")

    if not new_paths:
        print("Nothing new to ingest.")
        return

    print(f"Ingesting {len(new_paths)} file(s): {[os.path.basename(p) for p in new_paths]}")
    _, num_chunks = ingest_files(new_paths, vectorstore)
    if num_chunks > 0:
        print(f"Added {num_chunks} chunks to ./{os.path.dirname(DATA_DIR) or '.'}/chroma_db/")
    else:
        print("No new chunks were added (the files may contain no extractable text).")
    print("\nIngestion complete. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()

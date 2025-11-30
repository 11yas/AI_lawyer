import os
import re
import json
import hashlib
import pdfplumber
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from unstructured.partition.pdf import partition_pdf
    USE_UNSTRUCTURED = True
except ImportError:
    USE_UNSTRUCTURED = False

# === 1. Settings ===
DB_PATH = "./chroma_db"
LAWS_PATH = "./laws"
CACHE_PATH = "./cache"
COLLECTION_NAME = "laws"
BATCH_SIZE = 16
HASH_INDEX_FILE = os.path.join(CACHE_PATH, "file_hashes.json")

os.makedirs(LAWS_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)

# === Embedding model ===
EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)

# === Initialize Chroma ===
chroma = chromadb.PersistentClient(path=DB_PATH)


# === 2. Utility functions ===
def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def hash_file(path: str) -> str:
    """Create a stable hash for file contents."""
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def load_cache(fp: str):
    if os.path.exists(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_cache(fp: str, data):
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f)


def extract_text(path: str) -> str:
    """Try Unstructured.io first, fallback to pdfplumber."""
    if USE_UNSTRUCTURED:
        elements = partition_pdf(filename=path)
        text = "\n".join([el.text for el in elements if el.text])
        if text.strip():
            return text
    with pdfplumber.open(path) as pdf:
        pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
    return "\n".join(pages)


def split_text_semantically(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


# === 3. File hash management ===
def load_hash_index():
    if os.path.exists(HASH_INDEX_FILE):
        try:
            with open(HASH_INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_hash_index(index):
    with open(HASH_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


# === 4. Main processing ===
def _process_pdfs(collection, folder: str):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    if not files:
        print("⚠️  No PDF files found.")
        return

    file_hashes = load_hash_index()
    existing_ids = set(collection.get(include=[])["ids"]) if collection.count() else set()
    total_added = 0

    for file in files:
        path = os.path.join(folder, file)
        file_hash = hash_file(path)

        # Skip if same hash already processed
        if file_hashes.get(file) == file_hash:
            print(f"⏭️  Skipped '{file}' (unchanged).")
            continue

        print(f"\n📘 Processing file: {file}")

        try:
            text = extract_text(path)
            if not text.strip():
                print(f"⚠️  Skipped {file}: empty text.")
                continue

            chunks = split_text_semantically(clean_text(text))
            print(f"✂️  {len(chunks)} chunks generated.")

            to_add = []
            for chunk in tqdm(chunks, desc=f"→ {file}"):
                uid = hash_id(chunk + file)
                if uid in existing_ids:
                    continue

                cache_fp = os.path.join(CACHE_PATH, f"{uid}.json")
                emb = load_cache(cache_fp)
                if not emb:
                    emb = embedder.encode(chunk).tolist()
                    save_cache(cache_fp, emb)

                to_add.append({
                    "id": uid,
                    "text": chunk,
                    "emb": emb,
                    "meta": {"source": file}
                })

                if len(to_add) >= BATCH_SIZE:
                    _add_batch(collection, to_add)
                    total_added += len(to_add)
                    to_add = []

            if to_add:
                _add_batch(collection, to_add)
                total_added += len(to_add)

            # Save hash after success
            file_hashes[file] = file_hash
            save_hash_index(file_hashes)

            print(f"✅ {file}: added {len(chunks)} chunks.")
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    print(f"\n📊 Total new chunks added: {total_added}")


def _add_batch(collection, batch):
    try:
        collection.add(
            ids=[b["id"] for b in batch],
            documents=[b["text"] for b in batch],
            embeddings=[b["emb"] for b in batch],
            metadatas=[b["meta"] for b in batch],
        )
    except Exception as e:
        print(f"⚠️  Batch insert error: {e}")


# === 5. Public API ===
def load_pdfs(folder: str = LAWS_PATH):
    existing = [c.name for c in chroma.list_collections()]
    if COLLECTION_NAME in existing:
        collection = chroma.get_collection(COLLECTION_NAME)
        print(f"✅ Using existing collection '{COLLECTION_NAME}' ({collection.count()} items).")
    else:
        print(f"📦 Creating collection '{COLLECTION_NAME}'...")
        collection = chroma.create_collection(COLLECTION_NAME)
    _process_pdfs(collection, folder)
    print("🎉 Load complete.")
    return collection


def reload_pdfs(folder: str = LAWS_PATH):
    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)
        print(f"♻️  Old collection '{COLLECTION_NAME}' deleted.")
    collection = chroma.create_collection(COLLECTION_NAME)
    _process_pdfs(collection, folder)
    print(f"✅ Collection '{COLLECTION_NAME}' rebuilt.")
    return collection


if __name__ == "__main__":
    print("⚙️  Starting improved loader...")
    load_pdfs()

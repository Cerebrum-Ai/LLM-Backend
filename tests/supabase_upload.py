import os
from pathlib import Path
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ADMIN_KEY = os.environ.get("SUPABASE_ADMIN_KEY")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = 'test'

# Use admin key if available, else fallback to normal key
SUPABASE_AUTH_KEY = SUPABASE_ADMIN_KEY or SUPABASE_KEY

if not SUPABASE_URL or not SUPABASE_AUTH_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_ADMIN_KEY or SUPABASE_KEY in your environment.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_AUTH_KEY)

def upload_file(local_path: str, bucket: str = SUPABASE_BUCKET) -> str:
    file_path = Path(local_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")
    with open(file_path, "rb") as f:
        file_data = f.read()
    dest_path = file_path.name
    try:
        res = supabase.storage.from_(bucket).upload(dest_path, file_data, file_options={"content-type": "application/octet-stream"})
        if hasattr(res, "error") and res.error:
            raise RuntimeError(f"Upload failed: {res.error}")
        # Generate public URL
        public_url = supabase.storage.from_(bucket).get_public_url(dest_path)
        print(f"DEBUG: get_public_url({dest_path}) returned: {public_url}", flush=True)
        if not public_url:
            print("WARNING: Public URL is empty after upload, but upload may have succeeded.", flush=True)
            print(f"Uploaded to: [NO_PUBLIC_URL]", flush=True)
        else:
            print(f"Uploaded to: {public_url}", flush=True)
        return public_url
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        import traceback; traceback.print_exc()
        return ""

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter the path to the audio/image file to upload: ").strip()
    if not path or path.strip() in (".", ""):
        print("ERROR: No file path provided or path is invalid.", flush=True)
        sys.exit(1)
    try:
        url = upload_file(path)
        if url:
            print(f"Uploaded to: {url}", flush=True)
        else:
            print("ERROR: Upload failed or URL not returned.", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    path = input("Enter the path to the audio/image file to upload: ").strip()
    url = upload_file(path)
    print(f"Uploaded to: {url}")

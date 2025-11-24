import os
import shutil
import urllib.request
import zipfile
import tempfile
from tqdm import tqdm
from urllib.parse import urlparse

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file_override(url: str, dest_path: str):
    """
    Download a file from a URL and always overwrite the target.
    If it's a zip, extract its contents directly into dest_path (no extra folder level).
    If it's not a zip, save it directly to dest_path.
    """

    # Ensure parent dir for files
    dest_dir = dest_path if dest_path.endswith(('/', '\\')) else os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    # Temp file for download
    tmp_fd, tmp_path = tempfile.mkstemp()
    os.close(tmp_fd)

    filename = os.path.basename(urlparse(url).path) or "downloaded.file"
    print(f"Downloading {filename}...")

    try:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=tmp_path, reporthook=t.update_to)

        if zipfile.is_zipfile(tmp_path):
            # Remove dest_path if exists
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            os.makedirs(dest_path, exist_ok=True)

            # Extract into temp folder first
            with tempfile.TemporaryDirectory() as tmp_extract_dir:
                with zipfile.ZipFile(tmp_path, 'r') as z:
                    z.extractall(tmp_extract_dir)

                if "cache" not in dest_path:
                    # Move *contents* of extracted folder into dest_path
                    # cache folders have a different structure, so we skip this step for them
                    for item in os.listdir(tmp_extract_dir):
                        src = os.path.join(tmp_extract_dir, item)
                        dst = os.path.join(dest_path, item)
                        if os.path.isdir(src):
                            shutil.move(src, dst)
                        else:
                            shutil.move(src, dst)
                else:
                    # processing for cache folders of structure like zoom_cache.zip -> zoom_cache/zoom_cache/*
                    # also hold some auto generated macos metadata.
                    # Move the entire extracted folder into dest_path
                    contents = [x for x in os.listdir(tmp_extract_dir) if not x.startswith('__MACOSX')]
                    if len(contents) == 1 and os.path.isdir(os.path.join(tmp_extract_dir, contents[0])):
                        # Flatten: Only one top-level dir, move its contents
                        only_dir = os.path.join(tmp_extract_dir, contents[0])
                        for item in os.listdir(only_dir):
                            shutil.move(os.path.join(only_dir, item), os.path.join(dest_path, item))
                    else:
                        # Usual: move everything as-is
                        for item in contents:
                            shutil.move(os.path.join(tmp_extract_dir, item), os.path.join(dest_path, item))

            print(f"Extracted zip contents into '{dest_path}'.")
        else:
            # Ensure parent dir exists
            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.move(tmp_path, dest_path)
            tmp_path = None
            print(f"Saved file to '{dest_path}'.")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

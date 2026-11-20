"""Source text run inside the snapshot-download subprocess so a paused download resumes.

huggingface_hub 1.x writes every download to ``<etag>.<random>.incomplete`` and never reuses it, so a download
that was paused (the process is stopped) restarted every file from zero and left the old partial behind
(FU-275). This text swaps the library's private ``_download_to_tmp_and_move`` for a wrapper: with no partial it
calls the original (fast path, xet storage); with a partial from an earlier run it keeps the largest one, drops
the rest, cuts a safety margin off the end (the storage client may write ahead of what was flushed) and fetches
the remainder with an HTTP range request, checking the finished file against its SHA-256 name when it has one.
It runs as a string because the subprocess starts with ``python -c`` and cannot import this package. If the
library moves the private function, the wrapper is simply not installed and downloads behave as before.
"""

HF_RESUME_PATCH = '''
import glob, hashlib, os
try:
    import huggingface_hub.file_download as _fd
    _orig_dl = _fd._download_to_tmp_and_move
except Exception:
    _fd = _orig_dl = None
_SAFETY = 32 * 1024 * 1024
def _resumable(incomplete_path, destination_path, url_to_download, headers, expected_size, filename, force_download, etag, xet_file_data, tqdm_class=None):
    if destination_path.exists() and not force_download:
        return
    args = (incomplete_path, destination_path, url_to_download, headers, expected_size, filename, force_download, etag, xet_file_data, tqdm_class)
    stem = os.path.splitext(str(incomplete_path))[0]
    parts = [p for p in glob.glob(glob.escape(stem) + ".*incomplete") if os.path.isfile(p)]
    best = max(parts, key=os.path.getsize, default=None)
    for p in parts:
        if p != best:
            try: os.unlink(p)
            except OSError: pass
    size = os.path.getsize(best) if best else 0
    if best is None or size <= _SAFETY or (expected_size is not None and size > expected_size):
        if best:
            try: os.unlink(best)
            except OSError: pass
        return _orig_dl(*args)
    if best != str(incomplete_path):
        os.replace(best, incomplete_path)
    keep = size - _SAFETY
    with open(incomplete_path, "r+b") as f:
        f.truncate(keep)
        f.seek(keep)
        _fd.http_get(url_to_download, f, headers=headers, expected_size=expected_size, resume_size=keep, tqdm_class=tqdm_class)
    name = destination_path.name
    if len(name) == 64 and all(c in "0123456789abcdef" for c in name):
        digest = hashlib.sha256()
        with open(incomplete_path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != name:
            os.unlink(incomplete_path)
            raise OSError("Resumed download of " + filename + " failed its checksum; it will restart from zero.")
    _fd._chmod_and_move(incomplete_path, destination_path)
if _orig_dl is not None:
    _fd._download_to_tmp_and_move = _resumable
'''

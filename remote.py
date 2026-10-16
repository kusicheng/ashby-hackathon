"""Prepare and verify work on an existing authorized Jupyter server."""

import argparse
import json
import shutil
import time
import urllib.parse
from pathlib import Path

from src.jupyter_remote import JupyterServer, read_token
from src.remote_artifacts import checksum, create_bundle, extract_checked, verify_download
from src.utils import PROJECT_ROOT

PROBE = """import json, os, shutil, subprocess, sys
print(json.dumps({'python': sys.version, 'cwd': os.getcwd(), 'cpu_count': os.cpu_count(),
                  'disk_free_gb': round(shutil.disk_usage('.').free / 2**30, 1)}))
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
                     '--format=csv,noheader'], capture_output=True, text=True).stdout)
import torch
print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'bf16': torch.cuda.is_available() and torch.cuda.is_bf16_supported()}))
"""


def remote_path(value):
    path = str(value).replace("\\", "/")
    if (
        not path
        or path.startswith("/")
        or urllib.parse.unquote(path) != path
        or any(char in path for char in ":?#\r\n\x00")
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("Remote paths must be relative and cannot contain dot segments")
    return path


def upload_bundle(server, archive, directory):
    directory = remote_path(directory)
    # Never overwrite a previous project directory. Parent directories must exist.
    code = f"""import pathlib
p = pathlib.Path({directory!r})
if p.exists(): raise ValueError('Remote destination already exists')
p.mkdir(parents=True)
"""
    server.execute(code)
    remote_archive = directory + "/source.tar.gz"
    server.upload(archive, remote_archive)
    code = f"""import pathlib, tarfile, hashlib
p = pathlib.Path({remote_archive!r})
assert hashlib.sha256(p.read_bytes()).hexdigest() == {checksum(archive)!r}
destination = p.parent.resolve()
with tarfile.open(p) as archive:
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
            raise ValueError('Unsafe source bundle entry')
    archive.extractall(destination, filter='data')
print('Verified bundle extracted to', destination)
"""
    return server.execute(code)


def download_generation(server, remote, destination):
    remote = remote_path(remote)
    code = f"""import pathlib, tarfile, hashlib, json
source = pathlib.Path({remote!r})
records = json.loads((source / 'checksums.json').read_text())
assert records
for name, expected in records.items():
    original = source / name
    target = original.resolve()
    assert target.is_relative_to(source.resolve()) and not original.is_symlink()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
archive_path = source.parent / (source.name + '.tar.gz')
with tarfile.open(archive_path, 'w:gz') as archive:
    for name in [*sorted(records), 'checksums.json']:
        path = source / name
        if path.is_symlink(): raise ValueError('Symlink in backup')
        if path.is_file(): archive.add(path, arcname=path.relative_to(source), recursive=False)
print(json.dumps({{'archive': archive_path.as_posix(),
                  'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest()}}))
"""
    result = json.loads("".join(server.execute(code, timeout=300)))
    local_archive = destination.with_name(destination.name + ".tar.gz")
    server.download(result["archive"], local_archive)
    if checksum(local_archive) != result["sha256"]:
        raise ValueError("Archive checksum mismatch after download")
    extract_checked(local_archive, destination)
    count = verify_download(destination)
    return {
        "destination": str(destination),
        "verified_files": count,
        "archive_sha256": result["sha256"],
    }


def sync_backups(server, remote_root, destination):
    remote_root = remote_path(remote_root)
    code = f"""import pathlib, json
root = pathlib.Path({remote_root!r})
paths = []
for fold in sorted(root.glob('*/*')):
    if fold.is_symlink() or not fold.is_dir(): continue
    generations = sorted(p for p in fold.glob('generation_*')
                         if p.is_dir() and not p.is_symlink() and not p.name.endswith('.partial'))
    paths.extend(p.as_posix() for p in generations[-2:])
print(json.dumps(paths))
"""
    generations = json.loads("".join(server.execute(code)))
    results = []
    for remote in generations:
        relative = Path(remote).relative_to(remote_root)
        local = destination / relative
        if local.exists():
            verify_download(local)
        else:
            results.append(download_generation(server, remote, local))
        digest = checksum(local / "checksums.json")
        code = f"""import pathlib, hashlib, json, time, os
generation = pathlib.Path({remote!r})
assert hashlib.sha256((generation / 'checksums.json').read_bytes()).hexdigest() == {digest!r}
record = {{'verified_at': time.time(), 'generation': generation.name, 'checksums_sha256': {digest!r}}}
path = generation.parent / 'laptop_verified.json'
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(record))
os.replace(temporary, path)
"""
        server.execute(code)
    # Prune only older verified generations beneath this dedicated destination.
    destination = destination.resolve()
    for fold in destination.glob("*/*"):
        if fold.is_symlink() or not fold.is_dir():
            continue
        generations = sorted(path for path in fold.glob("generation_*") if path.is_dir())
        if len(generations) > 2:
            for latest in generations[-2:]:
                verify_download(latest)
            for old in generations[:-2]:
                if old.is_symlink() or not old.resolve().is_relative_to(destination):
                    raise ValueError("Unsafe backup retention path")
                verify_download(old)
                shutil.rmtree(old)
                archive = old.with_name(old.name + ".tar.gz")
                if archive.is_symlink() or not archive.resolve().is_relative_to(destination):
                    raise ValueError("Unsafe backup archive path")
                archive.unlink(missing_ok=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://icrn.ncsa.illinois.edu/user/kcheng17/lab")
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe")
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--output", type=Path, default=Path("outputs/remote/source.tar.gz"))
    bundle.add_argument("--include-data", action="store_true")
    upload = sub.add_parser("upload")
    upload.add_argument("archive", type=Path)
    upload.add_argument("--directory", required=True)
    execute = sub.add_parser(
        "execute", help="Run a reviewed local Python script in a fresh owned kernel"
    )
    execute.add_argument("script", type=Path)
    execute.add_argument("--timeout", type=int, default=120)
    backup = sub.add_parser("backup")
    backup.add_argument("--remote-generation", required=True)
    backup.add_argument(
        "--destination",
        type=Path,
        default=Path(f"outputs/remote_backups/download_{time.time_ns()}"),
    )
    watch = sub.add_parser(
        "watch-backups", help="Pull and verify two generations; acknowledge laptop durability"
    )
    watch.add_argument("--remote-export-root", required=True)
    watch.add_argument("--destination", type=Path, default=Path("outputs/remote_backups"))
    watch.add_argument("--interval", type=float, default=300)
    watch.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.command == "bundle":
        result = create_bundle(PROJECT_ROOT, args.output, args.include_data)
    else:
        server = JupyterServer(args.url, read_token(args.env))
        if args.command == "probe":
            result = server.execute(PROBE)
        elif args.command == "upload":
            result = upload_bundle(server, args.archive, args.directory)
        elif args.command == "execute":
            result = server.execute(args.script.read_text(encoding="utf-8"), timeout=args.timeout)
        elif args.command == "watch-backups":
            if not 1 <= args.interval <= 600:
                parser.error("Backup polling interval must be between 1 and 600 seconds")
            while True:
                result = sync_backups(server, args.remote_export_root, args.destination)
                print(json.dumps(result), flush=True)
                if args.once:
                    return 0
                time.sleep(args.interval)
        else:
            result = download_generation(server, args.remote_generation, args.destination)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

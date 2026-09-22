"""Registration and lineage contracts for immutable raw GIS source snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen


_PATH_TOKEN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _validate_token(value: str, field: str) -> str:
    normalized = value.strip().lower()
    if not _PATH_TOKEN.fullmatch(normalized):
        raise ValueError(f"{field} must use lowercase letters, digits, hyphens, or underscores")
    return normalized


@dataclass(frozen=True)
class RawSourceManifest:
    """Minimal provenance for one local, immutable source snapshot.

    This deliberately records no checksum.  The raw file itself remains under
    ``data/raw/{provider}/{city}/{snapshot_id}``; all derived map products
    belong under ``data/normalized``.
    """

    provider: str
    city: str
    snapshot_id: str
    source_name: str
    source_url: str
    download_date: str
    bbox: tuple[float, float, float, float]
    source_crs: str
    license: str
    files: tuple[str, ...] = ()
    schema_version: str = "raw-source/v1"

    def validate(self) -> None:
        _validate_token(self.provider, "provider")
        _validate_token(self.city, "city")
        _validate_token(self.snapshot_id, "snapshot_id")
        required = (self.source_name, self.source_url, self.download_date, self.source_crs, self.license)
        if not all(isinstance(value, str) and value.strip() for value in required):
            raise ValueError("source name, URL, date, CRS, and license must be nonempty")
        if len(self.bbox) != 4 or not all(isinstance(value, (int, float)) for value in self.bbox):
            raise ValueError("bbox must contain four numeric values: min_lon, min_lat, max_lon, max_lat")
        min_lon, min_lat, max_lon, max_lat = self.bbox
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("bbox must satisfy min_lon < max_lon and min_lat < max_lat")
        if any(Path(item).name != item or not item.strip() for item in self.files):
            raise ValueError("raw file entries must be nonempty filenames, not paths")

    @property
    def relative_directory(self) -> Path:
        return Path(_validate_token(self.provider, "provider")) / _validate_token(self.city, "city") / _validate_token(self.snapshot_id, "snapshot_id")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        value = asdict(self)
        value["bbox"] = list(self.bbox)
        value["files"] = list(self.files)
        return value


def register_raw_source(*, root: str | Path, manifest: RawSourceManifest, source_path: str | Path | None = None) -> Path:
    """Create one raw snapshot directory and optionally copy a local source.

    Existing snapshot directories are rejected so that raw sources are not
    silently replaced.  A directory source (for example a Shapefile bundle)
    is copied as a bundle; a file source (for example PBF or GeoJSON) is copied
    into the snapshot directory unchanged.
    """

    manifest.validate()
    target = Path(root) / manifest.relative_directory
    if target.exists():
        raise FileExistsError(f"raw snapshot already exists: {target}")
    if source_path is not None:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"source path does not exist: {source}")
        source_names = _source_names(source)
        if manifest.files and tuple(source_names) != manifest.files:
            raise ValueError("manifest files do not match the supplied source path")
    else:
        source = None
        source_names = list(manifest.files)

    written_manifest = RawSourceManifest(
        provider=manifest.provider,
        city=manifest.city,
        snapshot_id=manifest.snapshot_id,
        source_name=manifest.source_name,
        source_url=manifest.source_url,
        download_date=manifest.download_date,
        bbox=manifest.bbox,
        source_crs=manifest.source_crs,
        license=manifest.license,
        files=tuple(source_names),
        schema_version=manifest.schema_version,
    )
    target.mkdir(parents=True)
    if source is not None:
        if source.is_dir():
            for child in source.iterdir():
                destination = target / child.name
                if child.is_dir():
                    shutil.copytree(child, destination)
                else:
                    shutil.copy2(child, destination)
        else:
            shutil.copy2(source, target / source.name)
    (target / "source_manifest.json").write_text(
        json.dumps(written_manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def _source_names(source: Path) -> list[str]:
    if source.is_file():
        return [source.name]
    return sorted(item.name for item in source.iterdir())


def load_raw_manifest(path: str | Path) -> RawSourceManifest:
    """Load and validate a ``source_manifest.json`` file."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = RawSourceManifest(
        provider=raw["provider"], city=raw["city"], snapshot_id=raw["snapshot_id"],
        source_name=raw["source_name"], source_url=raw["source_url"], download_date=raw["download_date"],
        bbox=tuple(raw["bbox"]), source_crs=raw["source_crs"], license=raw["license"],
        files=tuple(raw.get("files", ())), schema_version=raw.get("schema_version", "raw-source/v1"),
    )
    manifest.validate()
    return manifest


def download_and_register_raw_source(
    *, root: str | Path, manifest: RawSourceManifest, download_url: str, timeout_seconds: int = 120
) -> Path:
    """Download one declared source file, then register it as a raw snapshot.

    The network transfer occurs in a temporary directory. Therefore a failed
    transfer never leaves a partial file under ``data/raw``. The URL is both an
    explicit input and the value recorded in the source manifest.
    """

    if not isinstance(download_url, str) or not download_url.strip():
        raise ValueError("download_url must be nonempty")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    filename = Path(urlparse(download_url).path).name
    if not filename or Path(filename).name != filename:
        raise ValueError("download URL must end with a filename")
    with tempfile.TemporaryDirectory(prefix="gis-raw-download-") as temp_dir:
        destination = Path(temp_dir) / filename
        request = Request(download_url, headers={"User-Agent": "GIS-Concept-LLM/0.1 raw-source-download"})
        with urlopen(request, timeout=timeout_seconds) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
        downloaded_manifest = RawSourceManifest(
            provider=manifest.provider,
            city=manifest.city,
            snapshot_id=manifest.snapshot_id,
            source_name=manifest.source_name,
            source_url=download_url,
            download_date=manifest.download_date,
            bbox=manifest.bbox,
            source_crs=manifest.source_crs,
            license=manifest.license,
            schema_version=manifest.schema_version,
        )
        return register_raw_source(root=root, manifest=downloaded_manifest, source_path=destination)

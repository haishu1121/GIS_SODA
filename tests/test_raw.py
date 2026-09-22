import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gis_concept_llm.raw import RawSourceManifest, download_and_register_raw_source, load_raw_manifest, register_raw_source


class RawSourceTests(unittest.TestCase):
    def setUp(self):
        self.manifest = RawSourceManifest(
            provider="osm", city="beijing", snapshot_id="osm-2026-09-20",
            source_name="OpenStreetMap", source_url="https://example.invalid/osm", download_date="2026-09-20",
            bbox=(116.30, 39.85, 116.50, 40.00), source_crs="EPSG:4326", license="ODbL-1.0",
        )

    def test_registering_a_file_copies_it_and_writes_a_lightweight_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "raw"
            source = Path(temp_dir) / "beijing.osm.pbf"
            source.write_bytes(b"not a real pbf")
            target = register_raw_source(root=root, manifest=self.manifest, source_path=source)
            self.assertEqual((target / "beijing.osm.pbf").read_bytes(), b"not a real pbf")
            saved = load_raw_manifest(target / "source_manifest.json")
            self.assertEqual(saved.files, ("beijing.osm.pbf",))
            self.assertNotIn("sha", saved.to_dict())

    def test_existing_snapshot_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            register_raw_source(root=temp_dir, manifest=self.manifest)
            with self.assertRaises(FileExistsError):
                register_raw_source(root=temp_dir, manifest=self.manifest)

    def test_invalid_bbox_is_rejected(self):
        invalid = RawSourceManifest(
            provider="osm", city="beijing", snapshot_id="bad", source_name="OpenStreetMap",
            source_url="https://example.invalid", download_date="2026-09-20", bbox=(2, 3, 1, 4),
            source_crs="EPSG:4326", license="ODbL-1.0",
        )
        with self.assertRaises(ValueError):
            invalid.validate()

    @patch("gis_concept_llm.raw.urlopen")
    def test_download_then_register_uses_the_declared_url_and_leaves_no_temp_file(self, mocked_urlopen):
        response = io.BytesIO(b"mock pbf")
        response.__enter__ = lambda: response
        response.__exit__ = lambda *args: None
        mocked_urlopen.return_value = response
        with tempfile.TemporaryDirectory() as temp_dir:
            target = download_and_register_raw_source(
                root=temp_dir, manifest=self.manifest,
                download_url="https://example.invalid/beijing-latest.osm.pbf",
            )
            self.assertEqual((target / "beijing-latest.osm.pbf").read_bytes(), b"mock pbf")
            self.assertEqual(load_raw_manifest(target / "source_manifest.json").source_url, "https://example.invalid/beijing-latest.osm.pbf")
        mocked_urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()

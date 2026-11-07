"""Generated synthetic rows only; no observed catalogue redistribution."""
import importlib.util
import csv
import json
import tempfile
import xml.etree.ElementTree as ET
from unittest.mock import patch
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('gcns_ingest', Path(__file__).resolve().parents[2] / 'tools/gcns_ingest.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Ingest(unittest.TestCase):
    def row(self, **kw):
        d = dict(source_id='9007199254740993',ra='0',dec='0',dist_1='.09',dist_16='.099',dist_50='.1',dist_84='.101',gcns_prob='.8',parallax='10',parallax_error='.1',phot_g_mean_mag='5')
        return d | kw
    def test_exact_id_and_reference_flux(self):
        ident,x,y,z,proxy = m.convert_row(self.row())
        self.assertEqual(ident,9007199254740993)
        self.assertAlmostEqual(x/m.PC,100)
        self.assertEqual((y,z),(0,0))
        self.assertAlmostEqual(proxy*(m.AU/x)**2,.01)
    def test_right_handed_axes_and_roundtrip(self):
        for ra,dec in [(0,0),(90,0),(0,90),(123.4,-56.7)]:
            _,x,y,z,_=m.convert_row(self.row(ra=str(ra),dec=str(dec)))
            self.assertAlmostEqual(math.degrees(math.asin(z/math.sqrt(x*x+y*y+z*z))),dec)
            if abs(dec)<90:self.assertAlmostEqual(math.degrees(math.atan2(y,x))%360,ra)
    def test_invalid_and_missing(self):
        for kw in [dict(source_id='1.2'),dict(dist_50='nan'),dict(dist_16='.2'),dict(dec='91'),dict(gcns_prob='.1'),dict(dist_50='1e308',dist_84='1e308')]:
            with self.assertRaises(ValueError):m.convert_row(self.row(**kw))
        self.assertIsNone(m.convert_row(self.row(phot_g_mean_mag='')))

    def test_atomic_schema_count_manifest_and_retained_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); row=self.row(); row.update({f'synthetic_column_{i}':'' for i in range(74-len(row))})
            raw=root/'gcns-main.csv'
            with raw.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            original=raw.read_bytes()
            xml=ET.Element('VOTABLE');table=ET.SubElement(xml,'TABLE')
            for key in ['column_name','unit']:ET.SubElement(table,'FIELD',name=key)
            data=ET.SubElement(table,'TABLEDATA')
            units=dict(ra='deg',dec='deg',dist_50='kpc',dist_16='kpc',dist_84='kpc',parallax='mas',phot_g_mean_mag='mag')
            for name in row:
                tr=ET.SubElement(data,'TR');ET.SubElement(tr,'TD').text=name;ET.SubElement(tr,'TD').text=units.get(name)
            schema=root/'schema.vot';ET.ElementTree(xml).write(schema)
            with patch.object(m,'ROWS',1):
                manifest=m.ingest(root)
                self.assertEqual(manifest['eligible_rows'],1)
                self.assertEqual(raw.read_bytes(),original)
                binary=(root/'gcns-v1.bin').read_bytes();self.assertEqual(len(binary),64)
                self.assertEqual(json.loads((root/'manifest.json').read_text())['raw_rows'],1)
                with self.assertRaises(ValueError):m.ingest(root,expected_sha='bad')
                schema.write_text('<VOTABLE/>')
                with self.assertRaises(ValueError):m.ingest(root)
                self.assertEqual((root/'gcns-v1.bin').read_bytes(),binary)
                with self.assertRaises(FileExistsError):m.fetch('https://invalid.invalid',raw)

if __name__=='__main__':unittest.main()

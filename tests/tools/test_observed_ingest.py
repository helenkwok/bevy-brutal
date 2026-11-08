"""All fixtures are synthetic; default tests never contact astronomy services."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('observed_ingest',Path(__file__).parents[2]/'tools/observed_ingest.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def vot(status='OK', ident='9007199254740993', unit='deg'):
 return f'<VOTABLE><RESOURCE><INFO name="QUERY_STATUS" value="{status}"/><TABLE><FIELD name="id" datatype="long"><VALUES null="-1"/></FIELD><FIELD name="ra" datatype="double" unit="{unit}"/><DATA><TABLEDATA><TR><TD>{ident}</TD><TD>0</TD></TR><TR><TD>-1</TD><TD>90</TD></TR></TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>'

def cns(**kw):
 r=dict(cns5_id='2',gaia_edr3_id='9007199254740993',ra='0',dec='0',epoch='2000',parallax='100',parallax_error='1',g_mag='NaN');r.update(kw);return r

def desi(**kw):
 r=dict(targetid='9007199254740993',ra='155',dec='0',ref_epoch='0',z='0.1',zerr='0.0001',zwarn='0',coadd_fiberstatus='0',zcat_primary='true',spectype='GALAXY',survey='main',program='bright');r.update(kw);return r

class ImportTests(unittest.TestCase):
 def load(self,text,**kw):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'fixture.vot';p.write_text(text);return m.load_table(p,**kw)
 def test_exact_id_and_sentinel(self):
  rows,fields=self.load(vot(),expected_rows=2,max_rows=2)
  self.assertEqual(rows[0]['id'],'9007199254740993');self.assertIsNone(rows[1]['id']);self.assertEqual(fields['ra']['unit'],'deg')
 def test_status_and_overflow_count(self):
  for text in [vot('OVERFLOW'),vot('ERROR'),vot().replace('<INFO name="QUERY_STATUS" value="OK"/>',''),vot()[:-10],vot().replace('<TD>90</TD>','')]:
   with self.assertRaises(ValueError):self.load(text)
  with self.assertRaises(ValueError):self.load(vot(),expected_rows=3)
  with self.assertRaises(ValueError):self.load(vot(),max_rows=1)
 def test_checksum_pin(self):
  self.load(vot(),expected_sha=hashlib.sha256(vot().encode()).hexdigest())
  with self.assertRaises(ValueError):self.load(vot(),expected_sha='0'*64)
 def test_cns_native_epoch_missing_g_and_snr(self):
  rec,stats=m.convert_cns5([cns(),cns(cns5_id='3',parallax_error='11')])
  self.assertEqual(len(rec),1);self.assertEqual(rec[0][0],2);self.assertEqual(rec[0][1],9007199254740993)
  self.assertAlmostEqual(rec[0][2]/m.PC,10);self.assertEqual(rec[0][5],2000);self.assertTrue(math.isnan(rec[0][-1]))
  self.assertEqual(stats['missing_direct_g'],2)
 def test_duplicate_or_bad_id_not_merged(self):
  with self.assertRaises(ValueError):m.convert_cns5([cns(),cns()])
  with self.assertRaises(ValueError):m.convert_desi([desi(),desi()])
  with self.assertRaises(ValueError):m.convert_desi([desi(targetid='9007199254740993.0')])
 def test_desi_primary_quality_and_unknown_epoch(self):
  rec,stats=m.convert_desi([desi(),desi(targetid='2',zcat_primary='false'),desi(targetid='3',zwarn='4')])
  self.assertEqual(len(rec),1);self.assertEqual(rec[0][0],9007199254740993);self.assertTrue(math.isnan(rec[0][5]));self.assertEqual(stats['dropped'],2)
 def test_cosmology_integral_not_cz(self):
  self.assertAlmostEqual(m.comoving_mpc(.1),418.4544876277,places=7)
  self.assertAlmostEqual(m.comoving_mpc(.5),1888.6253959334,places=7)
  for z in [-1,float('nan'),11]:
   with self.assertRaises(ValueError):m.comoving_mpc(z)
 def test_exact_id_crossmatch_does_not_certify_non_gaia_gaps(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'synthetic.csv';p.write_text('source_id\n9007199254740993\n3\n')
   stats=m.match_gcns([cns(),cns(cns5_id='3',gaia_edr3_id=None),cns(cns5_id='4',gaia_edr3_id='4')],p)
   self.assertEqual(stats['matched_rows'],1);self.assertEqual(stats['unresolved_rows'],2)
   self.assertEqual(stats['no_gaia_id'],1);self.assertEqual(stats['gcns_rows_scanned'],2)
   self.assertFalse(stats['merged'])
 def test_unknown_gaia_cannot_match_a_zero_gcns_id(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'synthetic.csv';p.write_text('source_id\n0\n')
   with self.assertRaises(ValueError):m.match_gcns([cns(gaia_edr3_id=None)],p)
 def test_units_and_query_contract(self):
  with self.assertRaises(ValueError):m.validate_schema('cns5',{'ra':{'unit':'rad'}})
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'fixture.vot';p.write_text(vot())
   with self.assertRaises(ValueError):m.check_query('cns5',p)
 def test_atomic_failure_retains_existing(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'diagnostic-v1.bin').write_bytes(b'keep')
   (p/'cns5-full.vot').write_text(vot());(p/'manifest.json').write_text(json.dumps({'raw_rows':2,'files':{'cns5-full.vot':{'sha256':'0'*64}}}))
   with self.assertRaises(ValueError):m.ingest('cns5',p)
   self.assertEqual((p/'diagnostic-v1.bin').read_bytes(),b'keep');self.assertFalse((p/'diagnostic-v1.bin.part').exists())
if __name__=='__main__':unittest.main()

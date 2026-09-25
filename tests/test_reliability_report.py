import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from reliability_report import aggregate,METHODS,METRICS


class AggregationTests(unittest.TestCase):
    def test_old_template_address_redirects_to_complete_page(self):
        folder=Path(__file__).resolve().parents[1]/'visualization'
        launcher=(folder/'reliability.template.html').read_text()
        self.assertIn('content="0;url=reliability.html"',launcher)
        self.assertNotIn('__STUDY_DATA__',launcher)
        self.assertTrue((folder/'templates/reliability.html.in').exists())

    def test_sample_sd_and_cell_assignment(self):
        records=[dict(count=12,noise=.03,metrics={m:{k:v for k in METRICS} for m in METHODS}) for v in (1,3,5)]
        cells=aggregate(records)
        filled=next(c for c in cells if c['count']==12 and c['noise']==.03)
        self.assertEqual(filled['repeats'],3)
        self.assertEqual(filled['values']['pinn']['velocity'],dict(mean=3.,sd=2.))
        empty=next(c for c in cells if c['count']==24 and c['noise']==.03)
        self.assertEqual(empty['repeats'],0)
        self.assertIsNone(empty['values']['pinn']['velocity']['mean'])

    def test_single_run_has_no_sample_sd(self):
        records=[dict(count=96,noise=.2,metrics={m:{k:7. for k in METRICS} for m in METHODS})]
        cell=next(c for c in aggregate(records) if c['count']==96 and c['noise']==.2)
        self.assertIsNone(cell['values']['analytical']['shear']['sd'])


if __name__=='__main__':unittest.main()

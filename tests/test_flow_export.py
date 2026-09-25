"""Le résultat doit pouvoir s'ouvrir sans les styles ou les données de Codex."""
import importlib.util
import json
import re
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FrameReader(HTMLParser):
    frame = None

    def handle_starttag(self, tag, attrs):
        if tag == 'iframe':
            self.frame = dict(attrs)['srcdoc']


class FlowExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('flow_builder', ROOT/'src/build_flow_visualization.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'index.html'
            module.build(ROOT/'results/inverse_pulsatile_20260913_165408_169589',
                         ROOT/'visualization/flow-fragment.template.html', output, standalone=True)
            cls.document = output.read_text(encoding='utf-8')
        reader = FrameReader()
        reader.feed(cls.document)
        cls.frame = reader.frame

    def test_standalone_has_utf8_and_complete_styles(self):
        self.assertIn('<meta charset="utf-8">', self.document)
        self.assertIn('<meta charset="utf-8">', self.frame)
        self.assertIn('Réalisation du bruit', self.frame)
        self.assertIn('--foreground: #1c3343', self.frame)
        self.assertNotIn('__FLOW_', self.document)

    def test_offline_scripts_and_all_experiments(self):
        self.assertNotRegex(self.frame, r'<script\b[^>]*\bsrc=')
        payload = re.search(r'<script type="application/json" id="pinn-flow-data">(.*?)</script>', self.frame, re.S)
        data = json.loads(payload[1])
        self.assertEqual({(c['noise'], c['seed']) for c in data['cases']},
                         {(n, s) for n in (0, .03, .1) for s in (42, 43, 44)})
        self.assertTrue(all(len(c['velocity']) == 51 for c in data['cases']))
        self.assertIn('d3', self.frame)

    def test_old_address_opens_the_finished_page(self):
        launcher = (ROOT/'visualization/tube-flow.template.html').read_text(encoding='utf-8')
        self.assertIn('content="0;url=index.html"', launcher)
        self.assertNotIn('__FLOW_', launcher)


if __name__ == '__main__':
    unittest.main()

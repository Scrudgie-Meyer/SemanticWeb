"""Regression tests for actual V1 failures, independent of V2 LLM output."""
import copy
import unittest
from pipeline import documents, read_json, validate_facts, BASE

class GroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = documents()
        cls.baseline = {f['id']: f for f in read_json(BASE / 'output' / 'baseline_v1' / 'facts.json')}

    def raw(self, key):
        fact = copy.deepcopy(self.baseline[key])
        fact.pop('id')
        return fact

    def test_known_invalid_facts_are_rejected(self):
        # Reversed roles, placeholder, unsupported relation, lost condition, type mismatch.
        for key in ['f015', 'f028', 'f038', 'f033', 'f051', 'f044', 'f006',
                    'f035', 'f048', 'f043', 'f003', 'f007']:
            with self.subTest(key=key):
                valid, rejected = validate_facts([self.raw(key)], self.docs, strict=True)
                self.assertFalse(valid)
                self.assertEqual(len(rejected), 1)

    def test_supported_positive_facts_survive(self):
        for key in ['f002', 'f018', 'f016', 'f009', 'f014']:
            with self.subTest(key=key):
                valid, rejected = validate_facts([self.raw(key)], self.docs, strict=True)
                self.assertEqual(len(valid), 1)
                self.assertFalse(rejected)

    def test_right_types_but_reversed_temporal_meaning_are_rejected(self):
        fact = self.raw('f018')
        fact['subject'], fact['object'] = fact['object'], fact['subject']
        valid, rejected = validate_facts([fact], self.docs, strict=True)
        self.assertFalse(valid)
        self.assertIn('wrong_direction', rejected[0]['reason'])

    def test_corrected_bearing_relation_is_supported(self):
        fact = self.raw('f028')
        fact['subject'].update(name='Діагностика ступичного підшипника', type='procedure')
        fact['object'].update(name='Заміна ступичного підшипника', type='procedure')
        valid, rejected = validate_facts([fact], self.docs, strict=True)
        self.assertEqual(len(valid), 1)
        self.assertFalse(rejected)

    def test_component_cannot_be_named_as_a_procedure(self):
        fact = self.raw('f028')
        fact['relation'] = 'CHECKED_BY'
        fact['subject'].update(name='Гул колеса', type='symptom')
        fact['object'].update(name='Ступичний підшипник', type='procedure')
        fact['evidence'] = 'Гул колеса перевіряють діагностикою ступичного підшипника.'
        valid, rejected = validate_facts([fact], self.docs, strict=True)
        self.assertFalse(valid)
        self.assertIn('type_name_mismatch', rejected[0]['reason'])

if __name__ == '__main__':
    unittest.main()

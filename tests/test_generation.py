import unittest

from soda.generation import TASKS, SPODGenerator
from soda.rewards import extract_answer


class GenerationTests(unittest.TestCase):
    def test_all_paper_interfaces_generate_valid_examples(self):
        examples = SPODGenerator(7).generate(1)
        self.assertEqual(len(examples), len(TASKS))
        for example in examples:
            example.validate()
            self.assertEqual(extract_answer(example.completion()), example.answer)


if __name__ == "__main__":
    unittest.main()

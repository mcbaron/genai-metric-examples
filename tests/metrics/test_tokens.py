import unittest

from promptx_core.metric.tokens import TokensMetric


class TestTokensMetric(unittest.TestCase):
    def setUp(self):
        self.metric = TokensMetric()

    def test_calculate(self):
        text = "This is a test sentence."
        num_tokens = self.metric(text)
        self.assertIsInstance(num_tokens, int)
        self.assertGreater(num_tokens, 0)

    def test_calculate_claude(self):
        text = "This is a test sentence."
        tools_json = ["tool1", "tool2"]
        try:
            num_tokens = self.metric.calculate_claude(text, tools_json)
            self.assertIsInstance(num_tokens, int)
            self.assertGreater(num_tokens, 0)
        except ImportError:
            self.skipTest("Anthropic library is not installed.")


if __name__ == "__main__":
    unittest.main()

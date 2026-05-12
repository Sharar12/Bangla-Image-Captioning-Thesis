import unittest

class TestAPIServer(unittest.TestCase):
    def test_imports(self):
        from src.api.server import app
        self.assertIsNotNone(app)

if __name__ == '__main__':
    unittest.main()

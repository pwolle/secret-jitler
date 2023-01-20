import jax
import unittest
import jax.random as jrn

from game import tests


class TestDummyHistory(unittest.TestCase):

    def test_works(self):
        # jit the function
        test_jit = jax.jit(tests.test_dummy_history, static_argnames=["player_total", "game_len"])

        # let it run once for faster loop afterwards
        key = jrn.PRNGKey(34527)
        test_jit(key=key)

        # test for 1000 random keys
        for i in range(10000):
            key = jrn.PRNGKey(8127346 - i)
            key, subkey = jrn.split(key)
            result = test_jit(key=subkey)
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()

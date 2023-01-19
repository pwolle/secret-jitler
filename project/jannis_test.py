import game
import random
from tqdm import tqdm

from game.run2 import *

import jax.random as jrn
import jax.numpy as jnp
import jax


def benchmark(i):
    key = jrn.PRNGKey(i)
    a = dummy_history(key)
    #return a
    
    
benchmark_jit = jax.jit(benchmark)

benchmark_jit(0)

print("jit done")

r = []

for i in tqdm(range(100000)):
    #ret = benchmark_jit(i)
    benchmark_jit(i)

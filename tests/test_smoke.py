import os
import sys
import torch
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_train import SimpleCNN, SimpleMLP
from jax_train import SimpleCNN as JCNN

def test_torch_cnn_shapes():
    m = SimpleCNN()
    assert m(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    assert sum(p.numel() for p in m.parameters()) == 421_834

def test_jax_cnn_shapes():
    v = JCNN().init(jax.random.PRNGKey(0), jnp.zeros((2, 28, 28, 1)), train=False)
    assert sum(x.size for x in jax.tree_util.tree_leaves(v["params"])) == 421_834

if __name__ == "__main__":
    test_torch_cnn_shapes()
    test_jax_cnn_shapes()
    print("Smoke tests passed successfully!")

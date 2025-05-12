import jax
import jax.numpy as jnp

class Metrics:

    @staticmethod
    def cosine_metric(x,y):
        x_norm = jnp.sqrt(jnp.sum(x**2,axis=-1))
        y_norm = jnp.sqrt(jnp.sum(y**2,axis=-1))
        sim = jnp.sum(x * y, axis=-1) / (x_norm * y_norm + 1e-8)
        return 1.0 - sim
    
    @staticmethod
    def euc_metric(x,y):
        return jnp.mean((x - y) ** 2)
    
    @staticmethod
    def log_euc_metric(x,y):
        k = 1 # Tune k
        return jnp.log(1 + k*jnp.mean((x - y) ** 2))
    
    @staticmethod
    def minkowski_metric(x,y):
        p = 2 # Tune p 
        return jnp.power(jnp.sum(jnp.power(jnp.abs(x-y), p)), 1/p)


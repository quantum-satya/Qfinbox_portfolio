# sampler_try.py
try:
    import neal
    sampler = neal.SimulatedAnnealingSampler()
    print("Using neal.SimulatedAnnealingSampler")
except Exception:
    try:
        # newer ocean SDK path
        from dwave.samplers import SimulatedAnnealingSampler
        sampler = SimulatedAnnealingSampler()
        print("Using dwave.samplers.SimulatedAnnealingSampler")
    except Exception:
        from dimod.reference.samplers import SimulatedAnnealingSampler
        sampler = SimulatedAnnealingSampler()
        print("Using dimod.reference.samplers.SimulatedAnnealingSampler")

# quick smoke test on a tiny Ising
h = {0: -1, 1: -1}
J = {(0,1): -1}
response = sampler.sample_ising(h, J, num_reads=100)
print(response.first)


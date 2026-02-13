"""
Generate test data for joint flow training.

This script creates small test datasets (1e5 events each) for:
- Flavor-tagged D → Ksππ (uses standard DKpp amplitude)
- CP-even combination (uses DKppCorrelated with cp=+1)
- CP-odd combination (uses DKppCorrelated with cp=-1)

All data is saved in Square Dalitz Plot (SDP) coordinates.
"""

import numpy as np
from DKpp import DKpp, DKppCorrelated, AmpSample
from Amplitude import SquareDalitzPlot2

# Number of events to generate
N_EVENTS = 100_000

# Initialize SDP object for coordinate transformation
dkpp = DKpp()
sdp_obj = SquareDalitzPlot2(dkpp.M(), dkpp.m1(), dkpp.m2(), dkpp.m3())


def dp_to_sdp(points_dp, sdp_obj, idx=(1, 2, 3)):
    """
    Convert Dalitz Plot coordinates to Square Dalitz Plot coordinates.

    Parameters
    ----------
    points_dp : ndarray, shape (N, 2)
        Points in DP coordinates (s12, s13)
    sdp_obj : SquareDalitzPlot2
        SDP object with mass definitions
    idx : tuple
        Particle indices (i, j, k)

    Returns
    -------
    ndarray, shape (N, 2)
        Points in SDP coordinates (m', θ')
    """
    i, j, k = idx
    s12 = points_dp[:, 0]
    s13 = points_dp[:, 1]
    mp = np.vectorize(lambda a, b: sdp_obj.MpfromM(a, b, i, j, k), otypes=[float])(s12, s13)
    tp = np.vectorize(lambda a, b: sdp_obj.TfromM(a, b, i, j, k), otypes=[float])(s12, s13)
    return np.column_stack([mp, tp])


def generate_flavor_data(n_events):
    """Generate flavor-tagged D → Ksππ events."""
    print(f"Generating {n_events:,} flavor-tagged events...")
    sampler = AmpSample(DKpp())
    points_dp = sampler.generate(n_events, nbatch=10000)
    points_sdp = dp_to_sdp(points_dp, sdp_obj)
    return points_sdp


def generate_cp_even_data(n_events):
    """Generate CP-even combination events."""
    print(f"Generating {n_events:,} CP-even events...")
    sampler = AmpSample(DKppCorrelated(cp=+1))
    points_dp = sampler.generate(n_events, nbatch=10000)
    points_sdp = dp_to_sdp(points_dp, sdp_obj)
    return points_sdp


def generate_cp_odd_data(n_events):
    """Generate CP-odd combination events."""
    print(f"Generating {n_events:,} CP-odd events...")
    sampler = AmpSample(DKppCorrelated(cp=-1))
    points_dp = sampler.generate(n_events, nbatch=10000)
    points_sdp = dp_to_sdp(points_dp, sdp_obj)
    return points_sdp


if __name__ == "__main__":
    print("=" * 60)
    print("Generating test data for joint flow training")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate all three datasets
    data_flavor = generate_flavor_data(N_EVENTS)
    data_even = generate_cp_even_data(N_EVENTS)
    data_odd = generate_cp_odd_data(N_EVENTS)

    # Save to files
    print("\nSaving data files...")
    np.save("D_Kspipi_SDP_test.npy", data_flavor)
    np.save("D_Kspipi_even_SDP_test.npy", data_even)
    np.save("D_Kspipi_odd_SDP_test.npy", data_odd)

    print("\nGenerated files:")
    print(f"  D_Kspipi_SDP_test.npy       - {len(data_flavor):,} flavor events")
    print(f"  D_Kspipi_even_SDP_test.npy  - {len(data_even):,} CP-even events")
    print(f"  D_Kspipi_odd_SDP_test.npy   - {len(data_odd):,} CP-odd events")

    # Print some statistics
    print("\nData statistics:")
    for name, data in [("Flavor", data_flavor), ("CP-even", data_even), ("CP-odd", data_odd)]:
        print(f"  {name}:")
        print(f"    m' range: [{data[:, 0].min():.4f}, {data[:, 0].max():.4f}]")
        print(f"    θ' range: [{data[:, 1].min():.4f}, {data[:, 1].max():.4f}]")

    print("\nDone!")

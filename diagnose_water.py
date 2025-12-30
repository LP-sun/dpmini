import numpy as np

print('=== O64H128 ===')
e1 = np.fromfile('collect/O64H128/energy.raw', dtype=np.float64)
print(f'Frames: {len(e1)}')
print(f'Energy: min={e1.min():.2f}, max={e1.max():.2f}, mean={e1.mean():.2f}')
print(f'Per-atom (192 atoms): {e1.mean()/192:.4f} eV/atom')

print('\n=== water ===')
e2 = np.fromfile('collect/water/energy.raw', dtype=np.float64)
print(f'Frames: {len(e2)}')
print(f'Energy: min={e2.min():.2f}, max={e2.max():.2f}, mean={e2.mean():.2f}')
print(f'Per-atom (192 atoms): {e2.mean()/192:.4f} eV/atom')
print(f'\nEnergy range ratio (water/O64H128): {(e2.max()-e2.min()) / (e1.max()-e1.min()):.2f}x')

# Check if water energies are unrealistic
print(f'\nWater single frame energies (first 10):')
print(e2[:10])

# Sources and assets

- Dynamic-bicycle equations and friction regressor: the autonomous-racing model accompanying [Trajectory Planning for Safe Dual Control with Active Exploration](https://arxiv.org/abs/2604.15507). The physical model uses seven states, explicit force and steering-rate inputs, and a fixed uncertain friction coefficient.
- Safety formulation: D. M. Cherenson, H. Lee, T. Kim and D. Panagou, [Distributionally Robust Safety Under Arbitrary Uncertainties: A Safety Filtering Approach](https://arxiv.org/abs/2605.12974), 2026.
- Barcelona track CSV, PyBullet vehicle and renderer: [uncertain-racecar-gym](https://github.com/joonlee16/uncertain-racecar-gym), revision `4af484c0e6f4336503e2cd8a3f34d581bd6b8fa8`. The renderer is adapted to the local track, state, and assets. Controllers and safety verification are implemented in this package.
- Browser racecar: Kenney Racing Kit asset supplied with `uncertain-racecar-gym`, under [CC0](../dual_gtk_drs_racing/assets/LICENSE_Kenney.txt). Display dimensions are fitted inside the collision footprint.
- Three.js and OrbitControls: revision r164, MIT, with the [license bundled locally](../dual_gtk_drs_racing/web/vendor/LICENSE-three.txt). The GLTF loader and geometry utilities are vendored from the same upstream revision.

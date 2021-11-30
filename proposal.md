# An Implicit Compressible SPH Solver for Snow Simulation


## Category & Topic

SPH solver for snow simulation. Snow has both plastic and elastic deformations, so realistic simulation requires solvers for both. A Lagrangian approach with Smoothed Particle Hydrodynamics (SPH) is used. Caution is needed at boundary conditions, since additional forces, such as adhesion and friction need to be simulated.

## Targets

- Minimal target
  - Elastic deformation solver
  - Boundary handling without friction and adhesion
  - Basic scene
- Desired target
  - Everything listed below
- Extended target
  - Improve on snow melting/disappearing

## Week 1

- Familiarise ourselves with Taichi Framework
- Elastic deformation solver
  - Implement basic solver: Bi-CGSTAB solver, matrix-free relaxed Jacobi solver
  - Setup attributes for each particle

## Week 2

- Elastic deformation solver
  - Use previously implemented solvers to solve for acceleration
  - Calculate velocity for each particle

## Week 3

- Plastic deformation solver
  - Calculate force for each particle
  - Euler's method to calculate position of the next particle
- Boundary handling
  - Read referenced papers on boundary handling

## Week 4

- Boundary handling
  - Implement adhesion forces and friction forces

## Week 5

- Scene setup
  - Create basic scenes: rolling balls, compression demonstration
  - Create advanced scenes
- Rendering
  - Estimate ~2 minutes/frame, at 50 frames/s, around 30s of footage, 50 hours of rendering
- Presentation preparation

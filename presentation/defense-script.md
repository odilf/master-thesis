1. Lagrangian formulation
2. Geometric view of Euler-Lagrange
3. Parallel algorithm
4. Forced systems
5. Lie groups

# Introduction

TODO

# Lagragnians

The presented work is based on the underlying observation that many physical systems can be described in the same framework. Think pendulums, robotic arms, electrons, black holes: all of them just need a configurafion space that can encode the possible states of the system, and a Lagrangian that defines how these states evolve. Namely, the possible paths of the system are those that *extremize* the integral of the Lagrangian, which is called the action.

To find these critical points we can take the integral,  take a variation of the path, set it to 0 and with integration parts we find this expressions. These are called the Euler-Lagrange equations, which characterize physical trajectories. These implicitly form differential equations, named the equations of motion for the system.

You may think that once we're here, we're done. And it is true that you have a set of ODEs that you can solve numerically, but we can do better. See, the Euler-Lagrange equations actually have some some geometric properties that we would really like to preserve. 

Let's use the pendulum as an example. The configuration space is a circle, S1. The Lagrangian acts on the tangent bundle, the phase space, so we also need a velocity. For each point, there is an R^1 degree of freedom. Therefore, the Lagrangian here acts on a cylinder. It takes every point and moves it along, forming a flow. The first property is that this flow is symplectic. Essentially, this means that it preserves areas and volumes, and also guarantees that there is so.e Hamiltonian that is preserved. We also have Noether's theorem, that states that for each symmetry, there is a conserved quantity. Here, rotational symmetry conserves angular momentum and time-symmetry (the Lagrangian doesn't change over time) conserves energy. 

These conservation laws, derived from the geometry, are a big deal. Solving the equations of motions doesn't guarantee any of this. What we can do instead is discretize the variational principle (extremization of the action) from which we derive the discrete Euler-Lagrange equations. Instead of TQ, we work on QxQ. There is a discrete Lagrangian that is equivalent to the continuous formulation. But crucially, we can approximate this discrete Lagrangian with some quadrature, and it is just as valid as any other Lagrangian. Which, in particular, is symplectic and conserves momentum maps. This is a well known but pretty clever trick, in my opinion. 



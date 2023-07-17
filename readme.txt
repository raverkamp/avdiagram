Roland Averkamp, 2017

This is a try at doing Enity Relationship Digram layout with linear programs.

The layout is described as constraints,
put table A left of table B
align table A and table C

Most times the constraints are binary, i.e they involve only two tables.

It is possible to group tables into rectangles.

No constraints to ensure tables do not overlap.

Or should one try with Gecode?

When using a full constraint solver the results might not be predictable.
The user does a small change, and suddenly everythin changes.

Routing: manually, but one can define points which a connection must touch.
These point can be defined by constraints as well.

If the system notes that some boxes overlap, add constraints?



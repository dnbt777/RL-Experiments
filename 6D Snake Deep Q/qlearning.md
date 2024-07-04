q learning optimizations that worked

main theme: compress states into smaller, info-dense chunks. maximize (info^2 / state space)

- gave it relative distances to objects from the head, rather than absolute positions of objects
- gave it log(distance) to things instead of the distance
- merged multiple 1/0 states into a single state by turning them into binary ints (i.e. 4-state (0, 1, 0, 0) => 1-state 0100) (now that I think abt it this doesnt reduce state space)
other:
- scaled rewards appropriately
- made learning rate smaller - having a 1% chance to randomly move has high p(killing the snake) if its running along a border or next to itself

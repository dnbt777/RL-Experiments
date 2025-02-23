
# rewrite with pytest


# unit should return -1, 0, or 1
from env import unit
assert unit(5) == 1
assert unit(2) == 1
assert unit(0) == 0
assert unit(-1) == -1
assert unit(-5) == -5
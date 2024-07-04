def unit(x):
    if x == 0:
        return 0
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


def abs_to_rel(head, abs_direction):
    return tuple(h - d for h, d in zip(head, abs_direction))

def rel_to_abs(head, rel_direction):
    return tuple(h + d for h, d in zip(head, rel_direction))

def notzero(x):
    if x == 0:
        return 1
    return x


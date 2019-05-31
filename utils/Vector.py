import math

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    
    def set_magnitude(self, m):
        prev_m = math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2))
        self.x *= (m / prev_m)
        self.y *= (m / prev_m)


def add(vector1, vector2):
    x = vector1.x + vector2.x
    y = vector1.y + vector2.y
    return Vector(x, y)


def sub(vector1, vector2):
    x = vector1.x - vector2.x
    y = vector1.y - vector2.y
    return Vector(x, y)


def divide(vector, n):
    return Vector(vector.x / n, vector.y / n)
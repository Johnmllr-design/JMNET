

def dot(self, a1, a2) -> float:
    if len(a1) != len(a2):
        return -1
    else:
        sum = 0.0
        for i in range(0, len(a1)):
            sum += a1[i] * a2[i]
        return sum
            
from math import isqrt


def is_prime(number):
    if type(number) is not int:
        raise TypeError("number must be an integer")
    if number < 2:
        return False
    if number == 2:
        return True
    if number % 2 == 0:
        return False
    for divisor in range(3, isqrt(number) + 1, 2):
        if number % divisor == 0:
            return False
    return True


def prime_flags(limit):
    if type(limit) is not int:
        raise TypeError("limit must be an integer")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    flags = bytearray([1]) * (limit + 1)
    flags[0] = 0
    if limit >= 1:
        flags[1] = 0
    for prime in range(2, isqrt(limit) + 1):
        if flags[prime]:
            for multiple in range(prime * prime, limit + 1, prime):
                flags[multiple] = 0
    return flags


def prime_factors(number):
    if type(number) is not int:
        raise TypeError("number must be an integer")
    if number < 1:
        raise ValueError("factorization requires a positive integer")
    factors = []
    divisor = 2
    while divisor * divisor <= number:
        while number % divisor == 0:
            factors.append(divisor)
            number //= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    if number > 1:
        factors.append(number)
    return factors


if __name__ == "__main__":
    flags = prime_flags(30)
    print("Primes through 30:", [number for number, flag in enumerate(flags) if flag])
    print("49 is prime:", is_prime(49))
    print("Factors of 84:", prime_factors(84))

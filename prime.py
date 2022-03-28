from collections import defaultdict
import functools


def factorial(num: int):
    ans = 1
    for i in range(1, num + 1):
        ans *= i
    return ans

def prime(num: int):
    if num <= 1: return False
    if num % 2 == 0: return False
    i = 3
    while i ** 2 <= num:
        if num % i == 0: 
            return False
        i += 2

    return True

def lowestPrimeFactor(num: int):
    if num == 1: return 1
    if num % 2 == 0: return 2
    i = 3
    while i ** 2 <= num:
        if num % i == 0: 
            return i
        i += 2

    return num

def primeFactors(num: int) -> list[int]:
    result = []

    while num > 1:
        lpf = lowestPrimeFactor(num)
        result.append(lpf)
        num //= lpf

    return result

def mostCommonNumber(nums: list[int]) -> int:
    max, ans = -1, [-1]
    seen = defaultdict(int)

    for num in nums:
        seen[num] += 1
    
    for key in seen.keys():
        if seen[key] > max:
            ans = [key]
            max = seen[key]
        elif seen[key] == max:
            ans.append(key)
            max = seen[key]

    return ans

def numZeroes(num):
    ans = 0
    num = str(num)

    for i in range(1, len(num)):
        if num[-i] != '0':
            break
        ans += 1

    return ans

zeroes = 0
for i in range(1500):
    fact = factorial(i)
    nz = numZeroes(fact)
    if nz > zeroes:
        print(nz, ' : ', i)
        zeroes = nz


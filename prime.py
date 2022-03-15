from collections import defaultdict


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

print(prime(3))    # true
print(prime(5))    # true
print(prime(10))   # false
print(prime(12))   # false
print(prime(31))   # true

print(primeFactors(10523)) # 2 2 263
nums = [0, 1, 2, 3, 4]
print(mostCommonNumber(nums))

print(nums[1:])
print(nums[:-1])
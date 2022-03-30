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

# convert mph pace to a 5k time
def paceToTime(pace: int):

    dist = 3.106 # miles in a 5k
    pace *= 1 / 60

    return dist / pace

def TimeToPace(time: int):
    dist = 3.106
    return dist / time * 60

print(TimeToPace(30))
print(TimeToPace(25))


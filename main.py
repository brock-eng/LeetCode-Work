class Solution(object):
    
    
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        rd = {
        'I' : 1,
        'V' : 5,
        'X' : 10,
        'L' : 50,
        'C' : 100,
        'D' : 500,
        'M' : 1000
        }
        # Loop through each char in string 
        # Retrieve the value for each char key
        # Add to running sum of values
        sum = 0
        for i in range(len(s) - 1):
            mult = 1 if (rd[s[i]] >= rd[s[i + 1]]) else -1
            sum += rd[s[i]] * mult

        sum += self.rd[s[-1]]

        return sum


def main():
    solution = Solution()
    num = solution.romanToInt('MCMXCIV')
    print(num)

if __name__ == "__main__": main()
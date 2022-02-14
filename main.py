from inspect import stack


class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution(object):
    
    # Given an integer, convert to roman numerals
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


    # Given two linked lists representing two nums in reverse order
    # add the two numbers together
    # Return as a similar reversed linked list
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        result = ListNode(0)
        resultTail = result
        carry = 0
        
        # while either list contains value or there is a carry value
        while l1 or l2 or carry:
            val1 = (l1.val if l1 else 0)
            val2 = (l2.val if l2 else 0)
            # divmod returns the whole num quotient and remainder 
            carry, out = divmod(val1 + val2 + carry, 10)
            
            # set next list node for result
            resultTail.next = ListNode(out)
            resultTail = resultTail.next
            
            # go to next list node if exists
            l1 = (l1.next if l1 else None)
            l2 = (l2.next if l2 else None)
        
        return result.next
        
    # multiply two strings
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        return (str(int(num1) * int(num2)))

    # Determine the length of the longest substring
    # without repeating characters
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s: return 0
        elif len(s) == 1: return 1

        used = {}

        maxl = start = 0

        for index , char in enumerate(s):
            if char in used and start <= used[char]:
                start = used[char] + 1
            else:
                maxl = max(maxl, index - start + 1)

            used[char] = index
        return maxl

    # Given a string of brackets, ex. "[]{}()()"
    # Determine if the string is valid
    # Valid string must properly close brackets
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        brackets = dict({
            '(' : ')',
            '[' : ']',
            '{' : '}'
        })
        bracketStack = list()

        for char in s:
            if char in brackets.keys():
                bracketStack.append(char)
            elif len(bracketStack) == 0 or char != brackets[bracketStack.pop()]:
                return False
        
        return len(bracketStack) == 0

    # Given two sorted lists, list1 and list2
    # Merge the two lists and return the sorted list
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        headNode = ListNode()
        nextNode = headNode
        while list1 != None or list2 != None:
            if list1.val > list2.val:
                nextNode.val = list2.val
                list2 = list2.next
            else:
                nextNode.val = list1.val
                list1 = list1.next
            
            newNode = ListNode()
            nextNode.next = newNode
            nextNode = newNode
        
        return headNode

    # Remove duplicates from array of nums in place
    # Return value is an int describing the total unique values
    def removeDuplicates(self, nums):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        k = i = 0
        lv = "init" # last checked value
        for i in nums:
            if i != lv:
                nums[k] = i
                lv = nums[k]
                k+=1
        
        return k
    
    # Remove duplicates from array of nums in place
    # Return value is an int describing the total unique values
    # Allow for multiple entries up to 'c' times
    def removeDuplicates2(self, nums, max):
        k = i = c = 0
        lv = "init" # last checked value
        for i in nums:
            if i != lv:
                nums[k] = i
                lv = i
                k+=1
                c=1
            elif c<max:
                nums[k] = i
                c+=1
                k+=1

        return k 
        
        # Reverse a linked list
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        
        prev = None
        curr = head
        while (curr):
            nextNode = curr.next
            curr.next = prev
            prev = curr
            curr = nextNode
            
        return prev

    # string 't' is generated by shuffling string 's' 
    # a new char will be added to string 't' in a random position
    # return the added character
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        # add the ASCII int value totals for each string
        # the added char will be found as t.chars - s.chars
        totalS = totalT = 0
        for char in s:
            totalS += ord(char)
        for char in t:
            totalT += ord(char)
        
        return chr(totalT - totalS)


    # Bullshit question from an interview
    def StringWeight(self, num):
        cMap = dict({'A' : 1})
        def CW(c):
            if c in cMap: return cMap[c]
            else:
                coeff = ord(c) - 64 + 1
                result = coeff * CW(chr(ord(c) - 1))
                cMap[c] = result
                return result

        rem = num
        ans = ''
        while rem > 0:
            char = 'A'
            while CW(char) <= rem:
                char = chr(ord(char) + 1)
            char = chr(ord(char) - 1)
            ans += char
            rem -= CW(char)

        return ans

    # Given a string s, return the longest palindromic substring in s
    def longestPalindrome_BAD(self, s):
        """
        :type s: str
        :rtype: str
        """
        def isPalindrome(ss):
            n = 0
            while n < len(ss)/2:
                if ss[n] != ss[-(n + 1)]:
                    return False
                n += 1
            return True

        # Init start/end index pointers
        # Start searching for minimum of palindrome - length 2
        # If found, start search again, increment length++
        # If not found, end search and return last found string
        size = len(s)
        if size == 0: return ''
        elif size == 1: return s

        l = 2
        while l <= size:      
            st = 0
            ed = st + l - 1
            while ed < size:
                print(s[st:(ed + 1)])
                if isPalindrome(s[st:(ed + 1)]):
                    ans = s[st:(ed + 1)]
                    break
                ed+=1
                st+=1
            l+=1
        
        return ans

    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        def helper(ss, l, r):
            while l >= 0 and r < len(s) and ss[l] == s[r]:
                l-=1
                r+=1
            return ss[l+1: r]

        ans = ''
        for i in range(len(s)):
            ans = max(helper(s, i, i), helper(s, i, i+1), ans, key=len)

        return ans

    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        def V(l, r, d):
            return min(l, r) * d

        if len(height) <= 1: return 0
        
        # start with left = 0 and right = (end of height)
        l = 0
        r = len(height) - 1
        ans = V(height[l], height[r], r - l)
        while l < r:
            if height[l] < height[r]:
                l+=1
                ans = max(V(height[l], height[r], r - l), ans)
            else:
                r-=1
                ans = max(V(height[l], height[r], r - 1), ans)
            
        
        
        return ans

def main():
    solution = Solution()
    
    height = [1,0,0,0,0,0,0,2,2]

    max = solution.maxArea(height)
    print(max)
    


if __name__ == "__main__": main()
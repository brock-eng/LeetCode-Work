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

def main():
    solution = Solution()

    nums = [1,2,2,3,4,5,5, 5, 5,6,6,7,7,7,7,7]
    intvalue = solution.removeDuplicates2(nums, 4)
    print(nums)
    print(intvalue)


if __name__ == "__main__": main()
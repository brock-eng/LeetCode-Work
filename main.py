from collections import defaultdict
from inspect import stack
import time

# Doubly linked list node
class DNode:
    def __init__(self, key, val):
        self.key, self.val = key, val
        self.prev = self.next = None

class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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
    def longestPalindrome_OLD(self, s):
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

    # Return the max area enclosed by two bars in a height map
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
                ans = max(V(height[l], height[r], r - l), ans)
        return ans

    # Find all possible text permutations given a dialed number
    def letterCombinations(self, digits: str) -> list[str]:
        digitsMap = {
            '0' : '', 
            '1' : '', 
            '2': 'abc', 
            '3': 'def', 
            '4': 'ghi', 
            '5': 'jkl', 
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv', 
            '9': 'wxyz'}
        
        if len(digits) == 0: return ""
        if len(digits) == 1: return list(digitsMap[digits])

        prev = self.letterCombinations(digits[:-1])
        next = digitsMap[digits[-1]]
        return [s + c for s in prev for c in next]
    
    # Given integer (n), return all combinations of 
    # well-formed parentheses 
    def generateParenthesis_OLD(self, n: int) -> list[str]:
        if not n: return []
        elif n == 1: return ["()"]
        single = "()"
        ans = ["()"]
        for i in range(n - 1):
            temp = list()
            for s in ans:
                for j in range(len(s)):
                    temp.append(s[:j] + single + s[j:])
            ans = temp

        return list(set(ans))

    # Given integer (n), return all combinations of 
    # well-formed parentheses 
    def generateParenthesis(self, n: int) -> list[str]:
        
        open = close = 0
        res = []
        def generate(open, close, s):
            if open == close == n:
                res.append(s)
            
            if open < n: 
                generate(open + 1, close, s + "(")
            if close < open:
                generate(open, close + 1, s + ")")
        generate(0, 0, "")
        return res

    # Search in a possible shifted sorted array for a value
    def search(self, nums: list[int], target: int) -> int:
        
        def M(l, r): return int((l + r) / 2)
        l = 0
        r = len(nums) - 1
        m = M(l, r)
        
        
        while l <= r:
            if nums[m] == target: return m
            
            if nums[l] <= nums[m]:
                if nums[l] <= target <= nums[m]:
                    r = m - 1
                    m = M(l,r)
                else:
                    l = m + 1
                    m = M(l,r)
            else:
                if nums[m] <= target <= nums[r]:
                    l = m + 1
                    m = M(l,r)
                else:
                    r = m - 1
                    m = M(l,r)
        
        return -1
        
    # find the start and ending indices of a target in sorted array
    def searchRange(self, nums: list[int], target: int) -> list[int]:
        
        if target > nums[-1] or target < nums[0]: return [-1, -1]

        # use binary search to find an instance of target
        # if found
            # start at found instance
            # while (start == instance) start--
            # while (end == instance) end++
        # else
            # return [-1, -1]
        def M(l,r): return (l + r)//2

        l = 0
        r = len(nums) - 1
        m = M(l,r)
        found = -1
        while l <= r:
            if nums[m] == target: 
                found = m
                break
                
            if target < nums[m]:
                r = m - 1
                m = M(l,r)
            else:
                l = m + 1
                m = M(l,r)

        if found > -1:
            l = r = found
            while nums[l] == target:
                l-=1
            while nums[r] == target:
                r+=1
            ans = [l + 1, r - 1]
        else:
            ans = [-1, -1]
            
        return ans
    
    # determine if an input num is a palindrome
    def isPalindrome(self, x: int) -> bool:
        digits = []
        while x > 0:
            digits.append(x % 10)
            x //= 10
        while len(digits) > 1:
            if digits.pop(0) != digits.pop():
                return False
        return True

    # add two ints represented as ints
    # do not convert to ints directly
    def addStrings(self, num1: str, num2: str) -> str:
        
        cn = lambda c : ord(c) - ord('0')

        r = 1
        carry = 0
        ans = ""
        val1 = val2 = 0
        while r <= len(num1) or r <= len(num2):
            if r <= len(num1):
                val1 = cn(num1[-r])
            else:
                val1 = 0
            if r <= len(num2):
                val2 = cn(num2[-r])
            else:
                val2 = 0
                
            sum = val1 + val2 + carry
            num, carry = sum % 10, sum >= 10
            
            ans = str(num) + ans
            r+=1
        ans = str(int(carry)) + ans
        return ans if ans[0] != '0' else ans[1:]

    # Return if a number is happy (true/false)
    # number is happy if the recursive sum of its
    # squared digits eventually reaches zero
    def isHappy(self, n: int) -> bool:
        def nextNum(n):
            sum = 0
            while n > 0:
                d = n % 10
                n = n // 10
                sum+= d * d
            return sum

        nf = nextNum(n)
        while nf != 1 and nf != n:
            nf = nextNum(nextNum(nf))
            n = nextNum(n)
            
        return nf == 1
    
    # Swap adjacent pairs in a linked list - return the head of the edited list
    def swapPairs(self, head) -> ListNode:
        
        if not head or not head.next: return head
        
        newstart = head.next.next
        head, head.next = head.next, head
        head.next.next = self.swapPairs(newstart)
        
        return head

    # divide two integers, can only use addition, subtraction, bit shifts
    # assume maximum return value is constrained by 32 bit system
    def divide(self, dividend: int, divisor: int) -> int:
        sign = 1 if (dividend > 0) is (divisor > 0) else -1
        
        dividend, divisor = abs(dividend), abs(divisor)
        res = 0
        while dividend >= divisor:
            
            curr, i = divisor, 1
            
            while curr <= dividend:
                dividend -= curr
                res += i
                
                curr <<= 1
                i <<= 1
        
        maxint = 2 ** 31 - 1
        return min(maxint, res) if sign == 1 else max(-maxint - 1, -res)

    # find all permutations for a given list of ints
    # assume all values are unique
    def permute(self, nums: list[int]) -> list[list[int]]:
        
        res = []
        def dfs(dec_space, fixed_path):
            if not dec_space: 
                res.append(fixed_path)
                return
            for i in range(len(dec_space)):
                dfs(dec_space[:i] + dec_space[i + 1:], fixed_path + [dec_space[i]])
        dfs(nums, [])

        return res

    # return all unique permutations given a list with possible
    # duplicate nums
    def permuteUnique(self, nums: list[int]) -> list[list[int]]:
        
        res = []
        def dfs(dec_space, fixed_path):
            if not dec_space: 
                res.append(fixed_path)
                return
            uniqueValues = set()
            for i in range(len(dec_space)):
                if dec_space[i] not in uniqueValues:
                    uniqueValues.add(dec_space[i])
                    dfs(dec_space[:i] + dec_space[i + 1:], fixed_path + [dec_space[i]])
        dfs(nums, [])

        return res

    # Given a matrix (2D array), 
    # rotate the matrix 90 degrees CW
    def rotate(self, matrix: list[list[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix[0])
        
        # UPL (0 + c, 0) -> (n, 0 + c)
        # UPR (0, n) -> (n, n)
        # LR  (n, n) -> (0, n)
        # LL  (0, 2) -> (0, 0)
        s = n - 1
        for row in range(n // 2 + n % 2):
            for col in range(n // 2):
                tmp = matrix[row][col]
                matrix[row][col] = matrix[col][s-row]
                matrix[col][s-row] = matrix[s - row][s - col]                
                matrix[s - row][s - col] = matrix[s - col][row]
                matrix[s - col][row] = tmp

    # given a list of words/strings, group all anagrams together
    # anagram -> eat, ate, tea
    # not anagram -> hello, hell, hi, hillo
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:

        res = []
        matches = dict()

        for s in strs:
            ss = ''.join(sorted(s))
            if ss not in matches.keys():
                matches[ss] = [s]
            else:
                matches[ss].append(s)

        for key in matches:
            res.append(matches[key])
        return res

    # Given an array of intervals where interval[i] = [start, end]
    # merge overlapping intervals
    def merge(self, intervals: list[list[int]]) -> list[list[int]]:

        # pseudo
        # [s1, e1] [s2, e2]
        # for each interval in intervals:
        #   if e1 > e2:
        #       add new interval
        #   elif e1 <= e2:
        #       merged previous interval

        if len(intervals) == 1: return intervals
        merged = []
        intervals.sort()
        
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        
        return merged

    # given a string (s) and a string (p), find all starting indices where
    # the substring defined by s[i:i + p_size] is an anagram of p
    def findAnagrams(self, s: str, p: str) -> list[int]:
        
        # edge case handling, init sizes and result array
        if not s or not p: return []
        p_size, s_size, res = len(p), len(s), []
        if p_size > s_size: return []
        
        # build a hashmap, 
        # init values for each char in p to += 1
        trackedChars = defaultdict(int)
        for c in p: 
            trackedChars[c] += 1
        
        # create initial window pass
        for c in s[0:p_size]:
            if c in trackedChars:
                trackedChars[c] -= 1

        # test for first case
        if all(value == 0 for value in trackedChars.values()):
            res.append(start)

        # iterate through the rest of the string s
        start = 0
        while start + p_size < s_size:
            start+=1
            # if a char exits the window, tracked -= 1
            if s[start - 1] in trackedChars:
                trackedChars[s[start - 1]] += 1
            # if a char enters the window, tracked += 1
            if s[start + p_size - 1] in  trackedChars:
                trackedChars[s[start + p_size - 1]] -= 1

            if all(value == 0 for value in trackedChars.values()):
                res.append(start)

        return res

    # Given an array of ints comprised of 1, 2, 3,
    # sort the array in place
    def sortColors(self, nums: list[int]) -> None:
        """
        Execute in place
        """

        # 0 -> left handed side
        # 1 -> placed in center
        # 2 -> right handed side

        # 0 count, 2 count -> l, r
        def swap(x, y):
            t = nums[x]
            nums[x] = nums[y]
            nums[y] = t

        # iterate through array
        l, r = 0, len(nums) - 1
        index = 0
        while index < r:
            if nums[index] == 0 and index > l:
                swap(index, l)
                l += 1
            elif nums[index] == 2 and index < r:
                swap(index, r)
                r -= 1
            else:
                index+=1

    # Convert a sorted array to a height balanced binary tree
    def sortedArrayToBST(self, nums: list[int]):
        if not nums: return
        if len(nums) == 1: return TreeNode(val=nums[0])
        else:
            mid = len(nums) // 2
            newNode = TreeNode(val = nums[mid],\
                left = self.sortedArrayToBST(nums[0:mid]),\
                right = self.sortedArrayToBST(nums[mid + 1: len(nums)]))
            return newNode

    # Traverse a binary tree inorder (left->right) and return
    # the result as an array
    def inorderTraversal(self, root) -> list[int]:
        res = []
        
        def helper(root):
            if root:
                helper(root.left)
                res.append(root.val)
                helper(root.right)
        
        helper(root)
        return res

    # Return a list of each level (top->bottom) in a binary tree
    def levelOrder(self, root) -> list[list[int]]:
        
        # input tree nodes
        # for each tree node, add left, right branch to list
        # add list to result
        # call function recursively for new list
        res = []
        def helper(treeList: list[TreeNode]):
            if len(treeList) == 0: return
            currvalues = [node.val for node in treeList]
            res.append(currvalues)
            
            nextNodes = []
            for node in treeList:
                if node.left: nextNodes.append(node.left)
                if node.right: nextNodes.append(node.right)
            helper(nextNodes)
            
        if root: helper([root])
        return res

    # Given an array of size n with nums in range [1, n-1]
    # find and return the duplicate num
    def findDuplicates(self, nums: list[int]) -> int:
        
        t = nums[0]
        h = nums[0]
        t, h = nums[t], nums[nums[h]]
        while t != h:
            t, h = nums[t], nums[nums[h]]
        t = nums[0]
        while t != h:
            t, h = nums[t], nums[h]
        return t

    # Build a binary tree from two integer arrays
    # representing the inorder and postorder traversals
    def buildTree(self, inorder: list[int], postorder: list[int]):
        if not inorder or not postorder: return None
        
        root = TreeNode()
        root.val = postorder.pop()
        inorderRoot = inorder.index(root.val)

        root.right = self.buildTree(inorder[inorderRoot + 1:], postorder)
        root.left = self.buildTree(inorder[:inorderRoot], postorder)

        return root

    # Build a binary tree from two integer arrays
    # representing the inorder and preorder traversals
    def buildTreePre(self, inorder: list[int], preorder: list[int]):
        if not inorder or not preorder: return None
        
        root = TreeNode()
        root.val = preorder.pop(0)
        inorderRoot = inorder.index(root.val)

        root.left = self.buildTreePre(inorder[:inorderRoot], preorder)
        root.right = self.buildTreePre(inorder[inorderRoot + 1:], preorder)

        return root

    # given a binary tree, populate each node with 
    # the node to the right as node.next
    def connect(self, root: 'Node') -> 'Node':

        if not root: return
        
        def helper(nodeList):
            if not nodeList: return
            nList = list()
            for node in nodeList:
                if node.left: nList.append(node.left)
                if node.right: nList.append(node.right)
                
            for index in range(len(nList) - 1):
                nList[index].next = nList[index + 1]
            helper(nList)
        
        helper([root])
        
        return root

    # Given an array nums where all nums appear 3 times except
    # one num, find the single occurrence
    def singleNumber(self, nums):
        
        a = b = 0
        for num in nums:
            a = (a ^ num) & ~b
            b = (b ^ num) & ~a

        return a | b

    # Given a linked list where each node points to a random
    # point in the list, create a hard/deep copy of the list
    def copyRandomList(self, head: Node):
        if not head: return
        
        # Copy linked list, intersperse into current list
        root = head
        copyHead = Node(root.val, next=root.next)
        copy = copyHead
        while head:
            head.next = copy
            head = copy.next
            if head:
                copy = Node(head.val, next=head.next)

        # assign random values to copied list
        head = root
        while head:
            copy = head.next
            if head.random:
                copy.random = head.random.next
            head = head.next.next

        # seperate lists
        head = root
        while head:
            copy = head.next
            head.next = copy.next
            if head.next:
                copy.next = head.next.next
            else:
                copy.next = None
            head = head.next

        return copyHead

    # given a linked list, return the node where a 
    # cycle begins, if applicable
    def detectCycle(self, head: ListNode):
        if not head: return
        
        slow, fast = head, head.next
        
        while True:
            if not (slow.next and fast.next and fast.next.next):
                return -1
            
            if slow == fast: 
                break
                
            slow = slow.next
            fast = fast.next.next
        
        root = head
        while root != slow:
            root = root.next
            slow = slow.next
        
        return root
        
    # Long ass question from hackerrank
    def projectEuler(self, num):
        def fac(n):
            res, i = n, 1
            while i < n:
                res *= (n - i)
                i+=1
            
            return res

           
        def sf(n):
            sum = 0
            while n > 0:
                sum += fac(n % 10)
                n //= 10
            
            return sum

        def sd(n):
            sum = 0
            while n >0:
                sum += n % 10
                n //= 10
            
            return sum

        def g(i):
            res = 1
            while sd(sf(res)) != i:
                res += 1
            
            return res

        def sg(i):
            return sd(g(i))
        
        return sg(num)

    # Reverse an integer (321 -> 123)
    def reverse(self, x: int) -> int:
        ans = []

        sign = 1 if x >= 0 else -1
        x = abs(x)

        while x > 0:
            digit = x % 10
            x //= 10
            ans.append(digit)
        
        rev = 0
        p = len(ans) - 1
        for num in ans:
            rev += num * 10**p
            p-=1

        rev *= sign
        rev = 0 if (rev > 2**31 - 1 or rev < -2**31) else rev
        return rev

    # Insertion sort a linked list
    def insertionSortList(self, head: ListNode) -> ListNode:       
        start = ListNode()
        curr = head
        
        while curr:
            prev, next = start, start.next
            
            while next and curr.val > next.val:
                prev = next
                next = next.next
            
            prev.next = curr
            d = curr.next
            curr.next = next
            curr = d
            
        return start.next

    # given a string of words, reverse the words in the string
    # return the result
    def reverseWords(self, s: str) -> str:
        
        # create empty list
        words = []
        
        # parse words from string and append into list
        words = s.strip().split(' ')
        
        
        # iterate through list in reverse order
        #   for each word, add to new string "result"
        words.reverse()
        ans = ' '.join(words)

        # output reversed string
        return ans

    # count the number of 1 bits in binary representation of
    # a number
    def countBits(self, num: int) -> int:
        count = 0
        while num > 0:
            if (1 & num):
                count += 1
            num >>= 1

        return count

    # Given two nums, return the fraction in string format
    def fractionToDecimal(self, n: int, d: int) -> str:
        # d [-n
        sign = '-' if n * d < 0 else ''
        n, d = abs(n), abs(d)
        i = n // d
        n %= d # remainder
        if not n: return str(i)
        ans = sign + str(i)

        seen = dict()
        dec = ""
        i = 0
        while n > 0:
            if n < d:
                n *= 10
            r = n // d
            n %= d
            dec += str(r)
            if n in seen:
                ans += "." + dec[:seen[n]] + "(" + dec[seen[n]:] + ")"
                return ans
            i += 1
            seen[n] = i

        ans += "." + dec
        return ans

    # given a 2d array of 1s and 0s that represents land (1)
    # and water (0), return number of islands
    def numIslands(self, grid: list[list[str]]) -> int:
        ans = 0
        m, n = len(grid), len(grid[0])
        
        seen = [False] * n * m
        def reduce(x, y):
            if x < 0 or x >= n or y < 0 or y >= m: return
            if grid[y][x] == '0': return
            if seen[x + y*n]: return

            seen[x + y*n] = True

            reduce(x+1, y)
            reduce(x, y+1)
            reduce(x-1, y)
            reduce(x, y-1)
        

        for y in range(m):
            for x in range(n):
                if grid[y][x] == "1" and not seen[x + y*n]:
                    reduce(x, y)
                    ans += 1
        
        return ans

def main():
    solution = Solution()
    nums = [1, 2, 4, 3, 5, 8, 7, 6, 4]
    num = 512
    start = time.perf_counter_ns()

    grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]

    # # # Put executable problems below
    ans = solution.numIslands(grid)
    print(ans)
    # # #
    end = time.perf_counter_ns()
    print("Execution time: ", (end - start)*1e-6, "[ms]")

if __name__ == "__main__": main()
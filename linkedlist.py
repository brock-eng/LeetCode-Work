class Node(object):
    def __init__(self):
        self.val = None
        self.next = None

class MyLinkedList(object):

    def __init__(self):
        self.head = Node()
        self.size = 0

    def get(self, index):
        """
        :type index: int
        :rtype: int
        """
        if index < 0 or index >= self.size + 1:
            return -1
        if not self.head:
            return -1
        
        curr = self.head
        for i in range(index + 1):
            curr = curr.next
            
        return curr.val

    def addAtHead(self, val):
        """
        :type val: int
        :rtype: None
        """
        newNode = Node()
        newNode.val = val
        newNode.next = self.head
        self.head = newNode
        self.size +=1

    def addAtTail(self, val):
        """
        :type val: int
        :rtype: None
        """
        curr = self.head
        for i in range(self.size - 1):
            curr = curr.next
        
        newNode = Node()
        curr.next = newNode
        newNode.val = val
        newNode.next = None
        self.size += 1
            

    def addAtIndex(self, index, val):
        """
        :type index: int
        :type val: int
        :rtype: None
        """
        curr = self.head
        for i in range(index):
            curr = curr.next
        
        newNode = Node()
        newNode.val = val
        newNode.next = curr.next
        curr.next = newNode
        
    def deleteAtIndex(self, index):
        """
        :type index: int
        :rtype: None
        """
        if index < 0 or self.size <= 0: return
        elif index == 1: 
            self.head = self.head.next
            self.size -= 1
            return
        
        prev = curr = self.head
        for i in range(index):
            prev = curr
            curr = curr.next
        
        prev.next = curr.next
        self.size-=1

    def print(self):
        curr = self.head
        s = "["
        for index in range(self.size):
            s += str(curr.val)
            if index != self.size - 1:
                s += ", "
            curr = curr.next
        s += "]"
        print(s)


def main():
    testList = MyLinkedList()
    testList.addAtHead(1)
    testList.addAtTail(3)
    testList.addAtIndex(1, 2)
    print(testList.get(1))
    testList.deleteAtIndex(1)
    print(testList.get(1))
    testList.print()

if __name__ == '__main__': main()
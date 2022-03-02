# Doubly linked list node
class DNode:
    def __init__(self, key, val):
        self.key, self.val = key, val
        self.prev = self.next = None

# customm LRUCache implmentation using hashmap and doubly linked list
class LRUCache:

    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = dict()

        self.left, self.right = DNode(0, 0), DNode(0, 0)
        self.left.next, self.right.prev = self.right, self.left


    def remove(self, node: DNode):
        next, prev = node.next, node.prev
        prev.next = next
        next.prev = prev

    def insert(self, node: DNode):
        prev, next = self.right.prev, self.right
        node.prev = prev
        node.next = next
        prev.next = next.prev = node

    def get(self, key: int) -> int:
        if key in self.cache:
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        n = DNode(key, value)
        self.cache[key] = n
        self.insert(n)

        if len(self.cache) > self.cap:
            # remove from list and delete lru from map
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]
from collections import defaultdict

# This is an implementation of a Trie data structure
# A Trie is a tree data structure which 
# stores valid char paths as children leading to stored strings.
# Useful for autocomplete, spellchecker, etc.

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.isEnd    = False

class Trie:

    def __init__(self):
        self.root = TrieNode()
     
    def insert(self, word: str) -> None:
        node = self.root
        
        for ch in word:
            node = node.children[ch]
        node.isEnd = True

    def search(self, word: str) -> bool:
        node = self.root

        for ch in word:
            if ch in node.children.keys():
                node = node.children[ch]
            else:
                return False
        
        return node.isEnd

    def startsWith(self, prefix: str) -> bool:   
        node = self.root

        for ch in prefix:
            if ch in node.children.keys():
                node = node.children[ch]
            else:
                return False
        return True

def main():
    test = Trie()
    test.insert("testword")
    print(test.search("testwor"))           # False
    print(test.search("testword"))          # True
    test.insert("testnewword")              
    print(test.startsWith("test"))          # True
    print(test.startsWith("testwx"))        # False
    return

if __name__ == "__main__": main()
#include <cmath>
#include <iostream>

using namespace std;

#define Print(x) std::cout << x << std::endl

typedef struct ListNode {
	int value;
	ListNode* nextNode = nullptr;
} ListNode;

ListNode* CreateLinkedList(int value)
{
	ListNode* head = (ListNode*)malloc(sizeof(ListNode));
	head->value = value;
	head->nextNode = nullptr;
	return head;
}

void AddListNode(ListNode* head, int newValue)
{
	while (head->nextNode)
	{
		head = head->nextNode;
	}
	ListNode* newNode = (ListNode*)malloc(sizeof(ListNode));
	newNode->value = newValue;
	newNode->nextNode = nullptr;
	head->nextNode = newNode;
	return;
}

bool InsertListNode(ListNode* head, int newValue, int atIndex)
{
	ListNode* curr = head;
	for (int i = 0; i < atIndex; i++)
	{
		if (!curr)
		{
			return false;
		}
		curr = curr->nextNode;
	}
	ListNode* newNode = (ListNode*)malloc(sizeof(ListNode));
	newNode->value = newValue;
	ListNode* tmp = curr->nextNode;
	curr->nextNode = newNode;
	newNode->nextNode = tmp;
	return true;
}

ListNode* ReverseLinkedList(ListNode* head)
{
	ListNode* next, * curr = head->nextNode, * tail = head;
	while (curr)
	{
		next = curr->nextNode;
		curr->nextNode = head;
		head = curr;
		curr = next;
	}
	tail->nextNode = nullptr;
	return head;
}

void PrintList(ListNode* head)
{
	ListNode* curr = head;
	Print("Printing Linked List:");

	while (curr != nullptr)
	{
		Print(curr->value);
		curr = curr->nextNode;
	}
	Print("Finished printing.");
}

void main() 
{
	int array[10];
	int i = 1;
	// 1 2 3 4 5 ... 10
	for (int j = 0; j < 10; j++)
	{
		array[j] = i++;
	}
	for (auto num : array)
	{
		cout << num << std::endl;
	}

	// Lesson 1:
	// Pointers!  Less go boy 
	int* startOfArrayPointer = &array[0];
	cout << "Memory Address: " << startOfArrayPointer << std::endl;
	// Dereference the pointer
	cout << "Value: " << *startOfArrayPointer << std::endl;

	// Lesson 2:
	// Pointer operations
	int* thirdIndexOfArrayPointer = startOfArrayPointer + 2;
	cout << "Third Value: " << *thirdIndexOfArrayPointer << std::endl;

	// Lesson 3:
	// Shit breaks down
	int* illegalPointer = &array[9];
	cout << "Final Value: " << *illegalPointer << std::endl;

	illegalPointer = illegalPointer + 1;
	cout << "??? Value: " << *illegalPointer << std::endl;
	// *illegalPointer = 5; // Pointing to the abyss
	// cout << "??? New Value: " << *illegalPointer << std::endl;

	// Lesson 4:
	// Linked Lists
	ListNode* head = CreateLinkedList(1);
	AddListNode(head, 2);
	AddListNode(head, 3);
	AddListNode(head, 4);
	AddListNode(head, 5);
	PrintList(head);
	head = ReverseLinkedList(head);
	PrintList(head);
}

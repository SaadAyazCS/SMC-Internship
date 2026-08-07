from stack import Stack
from queue import Queue
from linked_list import LinkedList

# -----------------------
# Stack
# -----------------------
print("----- STACK -----")

stack = Stack()

stack.push(10)
stack.push(20)
stack.push(30)

print("Top:", stack.peek())
print("Removed:", stack.pop())
print("Size:", stack.size())
print("Is Empty:", stack.is_empty())

print()

# -----------------------
# Queue
# -----------------------
print("----- QUEUE -----")

queue = Queue()

queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)

print("Front:", queue.front())
print("Removed:", queue.dequeue())
print("Size:", queue.size())
print("Is Empty:", queue.is_empty())

print()

# -----------------------
# Linked List
# -----------------------
print("----- LINKED LIST -----")

linked_list = LinkedList()

linked_list.insert(1)
linked_list.insert(2)
linked_list.insert(3)

print("Display:")
linked_list.display()

print("Search 2:", linked_list.search(2))

linked_list.delete(2)

print("After deleting 2:")
linked_list.display()
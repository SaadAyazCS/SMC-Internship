from stack import *
from my_queue import *
from linked_list import *

print("========== STACK ==========")

stack = create_stack()

push(stack, 10)
push(stack, 20)
push(stack, 30)

display(stack)

print("Top:", peek(stack))
print("Popped:", pop(stack))

display(stack)

print("Double Stack:", double_stack(stack))
print("Even Stack:", even_stack(stack))
print("Sum:", sum_stack(stack))


print("\n========== QUEUE ==========")

queue = create_queue()

enqueue(queue, 5)
enqueue(queue, 15)
enqueue(queue, 25)

display(queue)

print("Front:", front(queue))
print("Dequeued:", dequeue(queue))

display(queue)

print("Multiply Queue:", multiply_queue(queue))
print("Filter Queue (>10):", filter_queue(queue))
print("Sum:", sum_queue(queue))


print("\n========== LINKED LIST ==========")

lst = create_list()

insert(lst, 2)
insert(lst, 4)
insert(lst, 6)
insert(lst, 8)

display(lst)

delete(lst, 4)

display(lst)

print("Search 6:", search(lst, 6))
print("Squared:", square_values(lst))
print("Even Values:", even_values(lst))
print("Sum:", sum_values(lst))
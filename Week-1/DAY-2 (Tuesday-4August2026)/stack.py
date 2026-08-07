from functools import reduce
from helper import multiplier


def create_stack():
    return []


def push(stack, item):
    stack.append(item)


def pop(stack):
    if stack:
        return stack.pop()
    return "Stack is Empty"


def peek(stack):
    if stack:
        return stack[-1]
    return "Stack is Empty"


def display(stack):
    print("Stack:", stack)


# Functional Programming

def double_stack(stack):
    double = multiplier(2)
    return list(map(double, stack))


def even_stack(stack):
    return list(filter(lambda x: x % 2 == 0, stack))


def sum_stack(stack):
    return reduce(lambda x, y: x + y, stack)
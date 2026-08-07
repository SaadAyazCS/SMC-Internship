from functools import reduce


def create_list():
    return []


def insert(lst, value):
    lst.append(value)


def delete(lst, value):
    if value in lst:
        lst.remove(value)


def search(lst, value):
    return value in lst


def display(lst):
    print("Linked List:", lst)


# Functional Programming

def square_values(lst):
    return list(map(lambda x: x * x, lst))


def even_values(lst):
    return list(filter(lambda x: x % 2 == 0, lst))


def sum_values(lst):
    return reduce(lambda x, y: x + y, lst)
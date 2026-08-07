from functools import reduce


def create_queue():
    return []


def enqueue(queue, item):
    queue.append(item)


def dequeue(queue):
    if queue:
        return queue.pop(0)
    return "Queue is Empty"


def front(queue):
    if queue:
        return queue[0]
    return "Queue is Empty"


def display(queue):
    print("Queue:", queue)


# Functional Programming

def multiply_queue(queue):
    return list(map(lambda x: x * 3, queue))


def filter_queue(queue):
    return list(filter(lambda x: x > 10, queue))


def sum_queue(queue):
    return reduce(lambda x, y: x + y, queue)
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, data):
        new_node = Node(data)

        if self.head is None:
            self.head = new_node
            return

        current = self.head

        while current.next:
            current = current.next

        current.next = new_node

    def display(self):
        if self.head is None:
            print("Linked List is empty")
            return

        current = self.head

        while current:
            print(current.data, end=" -> ")
            current = current.next

        print("None")

    def search(self, key):
        current = self.head

        while current:
            if current.data == key:
                return True
            current = current.next

        return False

    def delete(self, key):
        if self.head is None:
            return

        if self.head.data == key:
            self.head = self.head.next
            return

        current = self.head

        while current.next:
            if current.next.data == key:
                current.next = current.next.next
                return
            current = current.next
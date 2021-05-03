class People:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.greeting()

    def greeting(self):
        return "Hello, {name}".format(name=self.name)


def main():
    people = [
        People('Jenny')
    ]
    print(dir(people[0]))

if __name__ == '__main__':
    main()
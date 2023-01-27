import random


def validated_input(expected: dict):
    expected = {str(k): v for k, v in expected.items()}
    texts = ["try again"]

    while True:
        read = input()

        try:
            return expected[read]

        except KeyError:
            print("not valid")
            print(random.choice(texts))
            continue


out = validated_input({i: i for i in range(10)})

print("out", out)

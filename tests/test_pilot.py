from inspeqtor import hello


def test_hello():
    assert hello(user="inspeqtor") == "Hello, inspeqtor!"

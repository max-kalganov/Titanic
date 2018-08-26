try:
    print(1/0)
except ArithmeticError:
    print("error")
print("after try")
def calc(s):
    s = s.replace('×', '*')
    s = s.replace('÷', '/')

    nums = []
    ops = []
    num = ""

    i = 0
    while i < len(s):
        if s[i].isdigit():
            num += s[i]
        else:
            nums.append(int(num))
            ops.append(s[i])
            num = ""
        i += 1

    nums.append(int(num))

    ans = nums[0]

    i = 0
    while i < len(ops):
        if ops[i] == '+':
            ans = ans + nums[i+1]
        elif ops[i] == '-':
            ans = ans - nums[i+1]
        elif ops[i] == '*':
            ans = ans * nums[i+1]
        elif ops[i] == '/':
            ans = ans / nums[i+1]
        i += 1

    return ans


while True:
    x = input("enter: ")

    if x == "0":
        break

    try:
        print("ans =", calc(x))
    except:
        print("wrong input")
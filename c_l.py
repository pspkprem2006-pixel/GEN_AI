n=int(input("enter a number:"))
if n>0:
    print("positive number")
elif n<0:
    print("negative number")
else:
    print("Zero")
print("===============================")
for i in range(1,51):
    if i%2==0:
        print(i)

print("===============================")
m=int(input("enter num for multiplication table:"))
for i in range(1,11):
    print(m,"*",i,"=",m*i)
    
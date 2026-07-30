print("Hello, World!")
if 5>2:
    print("Five is greater than two!")
n=input("enter your name: ")
if n:
    print("Hello",n)
age=int(input("enter your age:"))
if age>=19:
    print("you will be ",age+1,"next year. Go Get a job bro!!")
else:
    print("you need to study hard Bro!!")

H="Python lang"
print(H[0])
print(H[::-1])
a=[1,2,3,4,5]
s_e=int(input("enter a number to search:"))
for i in a:
    if i==s_e:
        print("found ",i," at ",a.index(i))
        break
    else:
        i+=1
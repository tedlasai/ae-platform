import browser_builder
import tkinter as tk
import browser_inputs_builder
import constants


root = tk.Tk()
root.geometry('1600x900'), root.title('Data Browser')  # 1900x1000+5+5


def fun1():
    print(1)

def fun2():
    print(2)

b = browser_inputs_builder.Broswer_with_inputs(root)
#b = browser_builder.Browser(root)
b.init_functions()
b.buttons_builder("xx",fun1,5,5)
b.buttons_builder("yy",fun2,7,6)

print(b.imgSize)
root.mainloop()
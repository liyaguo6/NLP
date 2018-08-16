# -*- coding: utf-8 -*-
from tkinter import *

from time import *


def main():
    def sendMsg():
        strtime = "我：" + strftime("%Y-%m-%d %H:%M:%S", localtime()) + "\n"

        txtMsgList.insert(END, strtime, 'greencolor')

        # 0.0是0行0列到END，表示全部，END表示插入末端

        txtMsgList.insert(END, txtMsg.get('0.0', END))

        txtMsg.delete('0.0', END)

    def cancelMsg():
        txtMsg.delete('0.0', END)

    def sendMsgEvent(event):
        if event.keysym == "Return":
            sendMsg()

    tk = Tk()

    tk.title("python聊天室")

    # 创建frame容器

    frameLT = Frame(width=500, height=320)

    frameLC = Frame(width=500, height=150, bg="yellow")

    frameLB = Frame(width=500, height=30)

    frameRT = Frame(width=200, height=500)

    # 创建控件

    txtMsgList = Text(frameLT)

    # 配置标签tag的属性，第一个参数为tag名字，第2个参数为前景色，背景色为默认白色

    txtMsgList.tag_config("greencolor", foreground='#008C00')

    txtMsg = Text(frameLC)

    txtMsg.bind_all("<KeyPress-Return>", sendMsgEvent)

    btnSend = Button(frameLB, text="send", width=8, command=sendMsg)

    btnCancel = Button(frameLB, text="cancel", width=8, command=cancelMsg)

    # myImage = PhotoImage(file="12.jpg")

    label = Label(frameRT)

    # 窗体布局

    frameLT.grid(row=0, column=0, columnspan=2, padx=1, pady=5)

    frameLC.grid(row=1, column=0, columnspan=2)

    frameLB.grid(row=2, column=0, columnspan=2)

    frameRT.grid(row=0, column=2, rowspan=3, padx=5, pady=4)

    # 固定大小

    frameLT.grid_propagate(0)

    frameLC.grid_propagate(0)

    frameLB.grid_propagate(0)

    frameRT.grid_propagate(0)

    # 控件布局

    btnSend.grid(row=2, column=0)

    btnCancel.grid(row=2, column=1)

    label.grid()

    txtMsgList.grid()

    txtMsg.grid()

    # 主事件循环

    tk.mainloop()


main()

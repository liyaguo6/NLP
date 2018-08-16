from summary import Summarizer
import pynlpir
import tkinter as tk
import time
try:

    import ttk as ttk

    import ScrolledText

except ImportError:

    import tkinter.ttk as ttk

    import tkinter.scrolledtext as ScrolledText


class TkinterGUIExample(tk.Tk):

    def __init__(self, *args, **kwargs):
        """

        Create & set window variables.

        """
        tk.Tk.__init__(self, *args, **kwargs)

        self.summerize = Summarizer(length=70, maxSumarySize=2)

        self.title("Summerize")
        self.initialize()

    def initialize(self):
        """

        Set window layout.

        """

        self.grid()
        self.respond = ttk.Button(self, text='submit', command=self.get_response)
        self.respond.grid(column=0, row=2, columnspan=1, sticky='nesw', padx=1, pady=3)
        self.cancel = ttk.Button(self, text='clear', command=self.cancelMsg)
        self.cancel.grid(column=1, row=2, columnspan=1, sticky='nesw', padx=1, pady=3)
        self.usr_input = ttk.Frame(width=500, height=200)
        self.txtMsg = tk.Text(self.usr_input)

        self.usr_input.grid(column=0, row=1, columnspan=2, sticky='nesw')
        self.txtMsg.grid()

        self.conversation = ttk.Frame(width=500, height=100)
        self.converMsg = tk.Text(self.conversation)
        self.conversation.grid(column=0, row=0, columnspan=2, sticky='nesw')
        self.converMsg.grid()

    def cancelMsg(self):
        self.txtMsg.delete('0.0', tk.END)
        self.converMsg.delete('0.0', tk.END)
    def get_response(self):
        """
        Get a response from the chatbot and display it.
        """
        try:
            pynlpir.open()
            paragraph = self.txtMsg.get('0.0', "end")
            result = self.summerize.get_result(paragraph)
            self.converMsg.insert(
                tk.END, "Summerizer: " + str(result) + "\n"
            )
            self.summerize.result =[]
            pynlpir.close()
        except IndexError:
            time.sleep(2)
            self.cancelMsg()
            self.summerize.result = []



gui_example = TkinterGUIExample()

gui_example.mainloop()

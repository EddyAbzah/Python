from tkinter import ttk
from tkinter import *


class GameSettingsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Game Settings")

        self.list_number_of_players = [2, 2, 3, 4, 5]
        self.list_board_matrix = [3, 3, 4, 5, 6, 7, 8, 9, 10]
        self.list_symbols_in_series_for_a_win = [3, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.selected_players = IntVar(value=self.list_number_of_players[0])
        self.selected_board_size = IntVar(value=self.list_board_matrix[0])
        self.selected_symbols_for_win = IntVar(value=self.list_symbols_in_series_for_a_win[0])

        label_players = ttk.Label(root, text="Number of Players:")
        label_players.grid(row=0, column=0, padx=10, pady=10, sticky=W)
        dropdown_players = ttk.OptionMenu(root, self.selected_players, *self.list_number_of_players)
        dropdown_players.grid(row=0, column=1, padx=10, pady=10, sticky=W)

        label_board_size = ttk.Label(root, text="Board Size:")
        label_board_size.grid(row=1, column=0, padx=10, pady=10, sticky=W)
        dropdown_board_size = ttk.OptionMenu(root, self.selected_board_size, *self.list_board_matrix)
        dropdown_board_size.grid(row=1, column=1, padx=10, pady=10, sticky=W)

        label_symbols_for_win = ttk.Label(root, text="Symbols in Series for a Win:")
        label_symbols_for_win.grid(row=2, column=0, padx=10, pady=10, sticky=W)
        dropdown_symbols_for_win = ttk.OptionMenu(root, self.selected_symbols_for_win, *self.list_symbols_in_series_for_a_win)
        dropdown_symbols_for_win.grid(row=2, column=1, padx=10, pady=10, sticky=W)

        button_submit = ttk.Button(root, text="Submit", command=self.get_selected_values)
        button_submit.grid(row=3, column=0, columnspan=2, pady=10)

    def get_selected_values(self):
        print(f"Number of Players: {self.selected_players.get()}")
        print(f"Board Size: {self.selected_board_size.get()}")
        print(f"Symbols in Series for a Win: {self.selected_symbols_for_win.get()}")


if __name__ == "__main__":
    root = Tk()
    app = GameSettingsGUI(root)
    root.mainloop()

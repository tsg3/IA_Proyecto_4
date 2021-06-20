import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk

buttons = None
matrix = None
text_vars = None
correction_buttons = None
text_labels = None
changes = None
last_answer = None
root = None

class NeuralNetwork:
    def __init__(self, learning_rate, size_l1): # constructor
        self.l1_nodes = size_l1
        self.weights_l1 = 2 * np.random.random_sample((100, self.l1_nodes)) - 1
        self.weights_lo = 2 * np.random.random_sample(self.l1_nodes) - 1
        self.learning_rate = learning_rate
    
    def activation(self, x): # sigmoid function
        return 1 / (1 + np.exp(-x))

    def d_activation_d_x(self, x): # derivative of sigmoid
        return self.activation(x) * (1 - self.activation(x))

    def guess(self, inputs): # predictions
        l1 = np.dot(inputs, self.weights_l1) # hidden layer 1
        output_l1 = self.activation(l1)
        lo = np.dot(output_l1, self.weights_lo) # output layer
        return l1, output_l1, lo, self.activation(lo)

    def mse(self, result, target): # mean squared error
        return 0.5 * ((target - result) ** 2)

    def compute_gradients(self, inputs, target): # generalized delta rule process; 1x100, 1x1
        l1, output_l1, lo, output_lo = self.guess(inputs) # 1xsize, 1xsize, 1x1, 1x1

        d_error_d_output_lo = output_lo - target # 1x1
        d_output_lo_d_lo = self.d_activation_d_x(lo) # 1x1
        d_lo_d_weights_lo = output_l1 # 1xsize
        
        d_error_d_weights_lo = (d_error_d_output_lo * d_output_lo_d_lo) * d_lo_d_weights_lo # 1x3

        d_error_d_lo = d_error_d_output_lo * d_output_lo_d_lo # 1x1
        d_lo_d_output_l1 = self.weights_lo # 1xsize
        d_error_d_output_l1 = d_error_d_lo * d_lo_d_output_l1 # 1xsize
        d_output_l1_d_l1 = self.d_activation_d_x(l1) # 1xsize
        d_l1_d_weights_l1 = inputs # 1x100

        d_error_d_weights_l1 = np.dot(d_l1_d_weights_l1.reshape(100, 1), \
            (d_error_d_output_l1 * d_output_l1_d_l1).reshape(1, self.l1_nodes)) # 100xsize

        return d_error_d_weights_l1, d_error_d_weights_lo

    def back_propagation(self, d_error_d_weights_l1, d_error_d_weights_lo): # generalized delta rule
        self.weights_l1 -= self.learning_rate * d_error_d_weights_l1
        self.weights_lo -= self.learning_rate * d_error_d_weights_lo

    def train(self, training_vectors, training_targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            index = np.random.randint(iterations)
            inputs = training_vectors[index]
            target = training_targets[index]

            d_error_d_weights_l1, d_error_d_weights_lo = self.compute_gradients(inputs, target)
            self.back_propagation(d_error_d_weights_l1, d_error_d_weights_lo)

            if current_iteration % 100 == 0:
                cumulative_error = 0
                for i in range(100):
                    index_test = np.random.randint(iterations)
                    inputs_test = training_vectors[index_test]
                    target_test = training_targets[index_test]
                    _, _, _, prediction = self.guess(inputs_test)
                    error = self.mse(prediction, target_test)
                    cumulative_error += error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors

def generate_training_data(size_dataset):
    vectors_list = []
    targets_list = []

    for i in range(size_dataset):
        vector = np.random.binomial(1, (i % 11) / 10, size = 100)

        target = 0
        if i % 11 == 10:
            target = 1
            vectors_list.append(vector)
            targets_list.append(target)
            continue

        for cell in vector:
            target += cell

        if target > 50:
            target = 1
        else:
            target = 0

        vectors_list.append(vector)
        targets_list.append(target)

    return vectors_list, targets_list

def binary_not(x):
    return (x - 1) ** 2

def clean_prediction_widgets():
    global correction_buttons
    global text_labels

    text_labels[0].grid_forget()
    text_labels[1].grid_forget()
    text_labels[2].grid_forget()
    correction_buttons[0].grid_forget()
    correction_buttons[1].grid_forget()

def set_button_color(i, j, fill):
    global matrix
    global buttons
    global changes

    if changes:
        clean_prediction_widgets()
    changes = True

    if fill == 0:
        matrix[i][j] = 0
        buttons[i][j].config(bg = "#000000", activebackground = "#000000")
    elif fill == 1:
        matrix[i][j] = 1
        buttons[i][j].config(bg = "#FFFFFF", activebackground = "#FFFFFF")
    else:
        matrix[i][j] = binary_not(matrix[i][j])
        if (matrix[i][j] == 0):
            buttons[i][j].config(bg = "#000000", activebackground = "#000000")
        else: 
            buttons[i][j].config(bg = "#FFFFFF", activebackground = "#FFFFFF")

def set_all_color(mode):
    global matrix
    global buttons
    global changes

    if changes:
        clean_prediction_widgets()
    changes = True

    if mode == 0:
        for i in range(10):
            for j in range(10):
                matrix[i][j] = 0
                buttons[i][j].config(bg = "#000000", activebackground = "#000000")
    elif mode == 1:
        for i in range(10):
            for j in range(10):
                matrix[i][j] = 1
                buttons[i][j].config(bg = "#FFFFFF", activebackground = "#FFFFFF")
    else:
        for i in range(10):
            for j in range(10):
                matrix[i][j] = np.random.randint(2)
                if (matrix[i][j] == 0):
                    buttons[i][j].config(bg = "#000000", activebackground = "#000000")
                else: 
                    buttons[i][j].config(bg = "#FFFFFF", activebackground = "#FFFFFF")

def make_prediction():
    global text_vars
    global correction_buttons
    global text_labels
    global last_answer

    inputs = np.array(matrix).reshape(100)
    _, _, _, last_answer = net.guess(inputs)

    text_vars[0].set("The result is: " + str(last_answer))
    if last_answer > 0.5:
        text_vars[1].set("That's a bright image!")
    else:
        text_vars[1].set("That's a dark image!")
    text_vars[2].set("Was that the correct answer?")

    text_labels[0].grid(row = 5, column = 11, columnspan = 2, sticky = tk.W)
    text_labels[1].grid(row = 6, column = 11, columnspan = 2, sticky = tk.W)
    text_labels[2].grid(row = 8, column = 11, columnspan = 2, sticky = tk.W + tk.E)

    correction_buttons[0].grid(row = 9, column = 11)
    correction_buttons[1].grid(row = 9, column = 12)

def suggest_answer():
    inputs = np.array(matrix).reshape(100)
    target = 1
    if last_answer > 0.5:
        target = 0

    d_error_d_weights_l1, d_error_d_weights_lo = net.compute_gradients(inputs, target)
    net.back_propagation(d_error_d_weights_l1, d_error_d_weights_lo)

def correction(option):
    global correction_buttons
    global text_vars

    correction_buttons[0].grid_forget()
    correction_buttons[1].grid_forget()

    if option:
        text_vars[2].set("Perfect! I knew it!")
    else:
        text_vars[2].set("Well... I'll try better next time! :)")
        suggest_answer()

def quit_me():
    root.quit()
    root.destroy()

def gui():
    global matrix
    global buttons
    global text_vars
    global correction_buttons
    global text_labels
    global changes
    global root

    matrix = []
    buttons = []

    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", quit_me)
    root.title("Neural Network")

    matrix = [[0] * 10 for _ in range(10)]

    for i in range(10):
        buttons_temp = []
        for j in range(10):
            buttons_temp.append(tk.Button(root, command = lambda i = i, j = j, fill = -1 : set_button_color(i, j, fill)))
            buttons_temp[j].config(bg = "#000000", activebackground = "#000000", width = 2, height = 2)
            buttons_temp[j].grid(row = i, column = j)
        buttons.append(buttons_temp)

    root.grid_columnconfigure(10, minsize = 50)

    button_black = tk.Button(root, bg = "#ABABAB", text = "Set all pixels black", width = 35, height = 2, command = lambda mode = 0 : set_all_color(mode))
    button_white = tk.Button(root, bg = "#ABABAB", text = "Set all pixels white", width = 35, height = 2, command = lambda mode = 1 : set_all_color(mode))
    button_random = tk.Button(root, bg = "#ABABAB", text = "Set all pixels to random values", width = 35, height = 2, command = lambda mode = -1 : set_all_color(mode))
    button_prediction = tk.Button(root, bg = "#ABABAB", text = "Make prediction", width = 35, height = 2, command = make_prediction)

    button_black.grid(row = 0, column = 11, columnspan = 2, sticky = tk.W + tk.E)
    button_white.grid(row = 1, column = 11, columnspan = 2, sticky = tk.W + tk.E)
    button_random.grid(row = 2, column = 11, columnspan = 2, sticky = tk.W + tk.E)
    button_prediction.grid(row = 3, column = 11, columnspan = 2, sticky = tk.W + tk.E)

    correction_buttons = [None] * 2

    correction_buttons[0] = tk.Button(root, bg = "#ABABAB", text = "Yes", width = 2, height = 2, command = lambda option = True : correction(option))
    correction_buttons[1] = tk.Button(root, bg = "#ABABAB", text = "No", width = 2, height = 2, command = lambda option = False : correction(option))

    text_vars = [None] * 3

    text_vars[0] = tk.StringVar()
    text_vars[1] = tk.StringVar()
    text_vars[2] = tk.StringVar()

    text_labels = [None] * 3

    text_labels[0] = tk.Label(root, textvariable = text_vars[0])
    text_labels[1] = tk.Label(root, textvariable = text_vars[1])
    text_labels[2] = tk.Label(root, textvariable = text_vars[2])

    changes = False

    root.mainloop()

def main():
    global net

    if len(sys.argv) == 4:
        try:
            net = NeuralNetwork(float(sys.argv[1]), int(sys.argv[2]))
        except:
            print("\nThere was an error creating the neural network!\n")
            return

        size_dataset = int(sys.argv[3])
        vectors_list, targets_list = generate_training_data(size_dataset)

        training_error = net.train(vectors_list, targets_list, size_dataset)

        print("\nTraining finished!\n\n...Starting the GUI...\n")

        gui()

        print("...Stopping the GUI...\n")

        plt.plot(training_error)
        plt.xlabel("Iterations")
        plt.ylabel("Error for all training instances")

        root = os.path.dirname(__file__)
        path = os.path.abspath(os.path.join(root, "graphs"))
        if not os.path.exists(path):
            os.mkdir(path)

        filename = "/cumulative_error_lr_" + sys.argv[1] + \
            "_l1nodes_" + sys.argv[2] + "_dataset_" + str(size_dataset) + "_"
        n = 0
        while os.path.isfile(path + filename + str(n) + ".png"):
            n += 1

        plt.savefig(path + filename + str(n) + ".png")
        plt.close()

        print("Program succesfully finished! You can check the error graph of the training in the following file:\n")
        print("\t" + path + filename + str(n) + ".png")
    else:
        print("\nSyntax error! This is the correct syntax:\n")
        print("\t$: python3 <src_file> <learning_rate> <size_layer> <size_dataset>\n")

if __name__ == "__main__":
    main()
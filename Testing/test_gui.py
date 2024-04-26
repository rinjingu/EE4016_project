import tkinter as tk
from tkinter import messagebox
import test

def calculate_loss():
    user_id = user_id_entry.get()
    number_of_recommender = int(number_of_recommender_entry.get())
    try:
        average_rmse, average_mae, average_ndcg_loss = test.Loss_Cal(number_of_recommender, user_id)
        messagebox.showinfo("Loss Calculation Result", f'Average RMSE: {average_rmse}\nAverage MAE: {average_mae}\nAverage NDCG Loss: {average_ndcg_loss}')
    except Exception as e:
        messagebox.showerror("Error", str(e))

def get_ratings():
    user_id = user_id_entry.get()
    number_of_recommender = int(number_of_recommender_entry.get())
    try:
        predicted_ratings = test.get_ratings(user_id, number_of_recommender)
        ratings_str = "\n".join([f'Item ID: {rating[0]}, Predicted Rating: {rating[1]}' for rating in predicted_ratings])
        messagebox.showinfo("Ratings Result", ratings_str)
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()

user_id_label = tk.Label(root, text="User ID")
user_id_label.pack()
user_id_entry = tk.Entry(root)
user_id_entry.pack()

number_of_recommender_label = tk.Label(root, text="Number of Recommender")
number_of_recommender_label.pack()
number_of_recommender_entry = tk.Entry(root)
number_of_recommender_entry.pack()

loss_cal_button = tk.Button(root, text="Calculate Loss", command=calculate_loss)
loss_cal_button.pack()

get_ratings_button = tk.Button(root, text="Get Ratings", command=get_ratings)
get_ratings_button.pack()

root.mainloop()
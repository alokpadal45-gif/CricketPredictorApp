import tkinter as tk
from tkinter import messagebox
import webbrowser
import threading
import json
import urllib.parse
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
import csv
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from collections import deque
from dotenv import load_dotenv

# loading my .env file so i can use my google credentials safely
load_dotenv()


# i got these from google cloud console, keeping them in .env so they dont get exposed
CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
REDIRECT_URI  = "http://localhost:8080"
SCOPE         = "openid email profile"

# this will store whoever is logged in right now, starts empty
logged_in_user = {"email": None, "name": None}

# i'm using a list to keep all the match records in memory while app is running
match_history = []

# this just saves the last prediction so i can use it in the graph
last_prediction = {}

# using deque here because i only want to keep the last 5 predictions
# when a 6th one comes in, the oldest one gets removed automatically
prediction_queue = deque(maxlen=5)

# name of my csv file where all the data gets saved
CSV_FILE = "cricket_data.csv"


# this runs once at the start to create the csv file if it doesn't already exist
# i check first so i don't accidentally overwrite data that's already there
def setup_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Runs", "Overs", "Wickets", "PredictedScore"])


# when the app opens i load whatever was saved before into match_history
# this way the machine learning model has data to learn from right away
def load_csv():
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                match_history.append({
                    "runs":      float(row["Runs"]),
                    "overs":     float(row["Overs"]),
                    "wickets":   int(row["Wickets"]),
                    "predicted": float(row["PredictedScore"])
                })


# every time a prediction is made i save it to the csv using append mode
# append mode means it adds to the end, it won't delete existing rows
def save_to_csv(runs, overs, wickets, predicted):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([runs, overs, wickets, predicted])


# this class handles what happens when google redirects back to my app after login
# google sends a code in the url to localhost:8080 and this catches it
class OAuthCallbackHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        # breaking down the url to get the query part which has the code in it
        query  = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)

        if "code" in params:
            # saving the code so i can exchange it for a token later
            self.server.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Login successful! You can close this tab.</h2></body></html>")
        else:
            # if google didn't send a code something went wrong
            self.server.auth_code = None
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Login failed. Please try again.</h2></body></html>")

    # i overrode this just to stop server logs from printing in my terminal
    def log_message(self, format, *args):
        pass


# google gives a one time code first, i have to swap it for an actual access token
# i send my client id, secret and the code to google and it gives me the token back
def exchange_code_for_tokens(auth_code):
    token_url = "https://oauth2.googleapis.com/token"
    data = urllib.parse.urlencode({
        "code":          auth_code,
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri":  REDIRECT_URI,
        "grant_type":    "authorization_code"
    }).encode("utf-8")
    req = urllib.request.Request(token_url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


# once i have the access token i can ask google for the user's name and email
def get_user_info(access_token):
    url = "https://www.googleapis.com/oauth2/v2/userinfo"
    req = urllib.request.Request(url)
    # sending the token in the header so google knows who is asking
    req.add_header("Authorization", f"Bearer {access_token}")
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def start_google_login(login_window):
    # building the google login url with all the required parameters
    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth?"
        + urllib.parse.urlencode({
            "client_id":     CLIENT_ID,
            "redirect_uri":  REDIRECT_URI,
            "response_type": "code",
            "scope":         SCOPE,
            "access_type":   "offline",
            "prompt":        "consent"
        })
    )

    def oauth_flow():
        # starting a temporary server on port 8080 to wait for google's redirect
        server = HTTPServer(("localhost", 8080), OAuthCallbackHandler)
        server.auth_code = None
        server.timeout   = 120  # wait max 2 minutes for the user to log in

        # opening the browser so user can sign in
        webbrowser.open(auth_url)

        # this pauses here and waits for exactly one request from google
        server.handle_request()
        auth_code = server.auth_code
        server.server_close()

        if not auth_code:
            login_window.after(0, lambda: messagebox.showerror(
                "Login Failed", "Google authentication failed or was cancelled."))
            return

        try:
            tokens       = exchange_code_for_tokens(auth_code)
            access_token = tokens["access_token"]

            user_info = get_user_info(access_token)
            logged_in_user["email"] = user_info.get("email", "Unknown")
            logged_in_user["name"]  = user_info.get("name",  "User")

            # can't update tkinter from a background thread so i use after() to do it safely
            login_window.after(0, lambda: open_cricket_app(login_window))

        except Exception as e:
            login_window.after(0, lambda: messagebox.showerror(
                "Login Error", f"Failed to complete login:\n{e}"))

    # running the whole login flow in a background thread
    # this stops the login window from freezing while waiting for google
    threading.Thread(target=oauth_flow, daemon=True).start()


# this helper function calculates the predicted score for any runs/overs/wickets
# i separated it out so i can reuse it when building the graph
def calc_predicted(runs, overs, wickets):
    total_overs = 20
    if overs <= 0:
        return int(runs)
    run_rate          = runs / overs
    remaining_overs   = total_overs - overs
    adjusted_run_rate = run_rate * (1 - (wickets * 0.03))
    if adjusted_run_rate < 0:
        adjusted_run_rate = 0
    basic = runs + (adjusted_run_rate * remaining_overs)

    # only use machine learning if i have at least 3 records to train on
    if len(match_history) >= 3:
        X = np.array([[m["runs"], m["overs"], m["wickets"]] for m in match_history])
        y = np.array([m["predicted"] for m in match_history])
        mdl = LinearRegression()
        mdl.fit(X, y)
        ml_pred = mdl.predict([[runs, overs, wickets]])[0]
        result  = (basic + ml_pred) / 2
    else:
        result = basic

    return int(max(result, runs))


def open_cricket_app(login_window):
    # closing the login window before opening the main app
    login_window.destroy()
    setup_csv()
    load_csv()

    window = tk.Tk()
    window.title("Cricket Score Prediction System")
    window.geometry("420x560")
    window.resizable(False, False)

    # green welcome bar at the top showing who is logged in
    tk.Label(window,
             text=f"Welcome, {logged_in_user['name']}!  |  {logged_in_user['email']}",
             font=("Arial", 9, "italic"), fg="white", bg="#4CAF50",
             anchor="w", padx=10).pack(fill="x")

    tk.Label(window, text="Cricket Score Prediction System",
             font=("Arial", 14, "bold")).pack(pady=12)

    # using a frame to hold the input fields so i can use grid layout inside it
    frame = tk.Frame(window)
    frame.pack(pady=5)

    tk.Label(frame, text="Runs Scored:",  font=("Arial", 11)).grid(row=0, column=0, padx=10, pady=8, sticky="w")
    runs_entry = tk.Entry(frame, width=15)
    runs_entry.grid(row=0, column=1, padx=10, pady=8)

    tk.Label(frame, text="Overs Played:", font=("Arial", 11)).grid(row=1, column=0, padx=10, pady=8, sticky="w")
    overs_entry = tk.Entry(frame, width=15)
    overs_entry.grid(row=1, column=1, padx=10, pady=8)

    tk.Label(frame, text="Wickets Lost:", font=("Arial", 11)).grid(row=2, column=0, padx=10, pady=8, sticky="w")
    wickets_entry = tk.Entry(frame, width=15)
    wickets_entry.grid(row=2, column=1, padx=10, pady=8)

    # this label will update to show the prediction result after clicking predict
    result_label = tk.Label(window, text="", font=("Arial", 11), justify="left")
    result_label.pack(pady=10)

    def predict():
        try:
            runs    = float(runs_entry.get())
            overs   = float(overs_entry.get())
            wickets = int(wickets_entry.get())
            total_overs = 20

            # basic validation checks before doing any calculation
            if overs == 0:
                result_label.config(text="Please enter overs played.", fg="red"); return
            if overs > total_overs:
                result_label.config(text="Overs cannot exceed 20.", fg="red"); return
            if wickets < 0 or wickets > 10:
                result_label.config(text="Wickets must be between 0 and 10.", fg="red"); return

            run_rate          = runs / overs
            remaining_overs   = total_overs - overs

            # each wicket lost reduces the expected run rate by 3%
            # because losing wickets slows the team down
            adjusted_run_rate = run_rate * (1 - (wickets * 0.03))
            if adjusted_run_rate < 0:
                adjusted_run_rate = 0

            basic_predicted = runs + (adjusted_run_rate * remaining_overs)

            # if i have 3 or more records i mix in the ml prediction too
            # averaging both gives a more balanced and accurate result
            if len(match_history) >= 3:
                X = np.array([[m["runs"], m["overs"], m["wickets"]] for m in match_history])
                y = np.array([m["predicted"] for m in match_history])
                model = LinearRegression()
                model.fit(X, y)
                ml_predicted    = model.predict([[runs, overs, wickets]])[0]
                final_predicted = (basic_predicted + ml_predicted) / 2
                method          = "Machine Learning + Formula"
            else:
                final_predicted = basic_predicted
                method          = "Formula Only"

            # predicted score should never be less than runs already on the board
            final_predicted = max(final_predicted, runs)

            save_to_csv(runs, overs, wickets, int(final_predicted))
            match_history.append({"runs": runs, "overs": overs,
                                   "wickets": wickets, "predicted": int(final_predicted)})

            # saving to last_prediction so the graph function can use it
            last_prediction["runs"]      = runs
            last_prediction["overs"]     = overs
            last_prediction["wickets"]   = wickets
            last_prediction["predicted"] = int(final_predicted)

            prediction_queue.append(int(final_predicted))

            result_label.config(
                text=f"Run Rate      :  {run_rate:.2f} runs/over\n"
                     f"Predicted Score :  {int(final_predicted)} runs\n"
                     f"Method        :  {method}\n"
                     f"Records Saved :  {len(match_history)}",
                fg="green")

        except ValueError:
            result_label.config(text="Please enter valid numbers.", fg="red")

    def show_graph():
        if not last_prediction:
            messagebox.showinfo("No Data", "Please make at least one prediction first!")
            return

        runs        = last_prediction["runs"]
        overs       = last_prediction["overs"]
        wickets     = last_prediction["wickets"]
        total_overs = 20

        run_rate          = runs / overs
        adjusted_run_rate = run_rate * (1 - (wickets * 0.03))
        if adjusted_run_rate < 0:
            adjusted_run_rate = 0

        # building past data by spreading runs evenly across overs played
        # for example 90 runs in 10 overs = 9 runs per over
        past_overs = []
        past_runs  = []
        for o in range(1, int(overs) + 1):
            past_overs.append(o)
            past_runs.append(int(run_rate * o))

        # building future data from current over to over 20
        # adding adjusted run rate for each remaining over
        future_overs     = []
        future_predicted = []
        for o in range(int(overs), total_overs + 1):
            future_overs.append(o)
            extra_overs    = o - overs
            projected_runs = runs + (adjusted_run_rate * extra_overs)
            future_predicted.append(int(projected_runs))

        # pulling data from match history for chart 1 and chart 3
        runs_list      = [m["runs"]              for m in match_history]
        predicted_list = [m["predicted"]         for m in match_history]
        run_rate_list  = [m["runs"] / m["overs"] for m in match_history]

        # regression line calculation for chart 1
        X_reg = np.array(runs_list).reshape(-1, 1)
        y_reg = np.array(predicted_list)
        reg_model = LinearRegression()
        reg_model.fit(X_reg, y_reg)
        regression_line = reg_model.predict(X_reg)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Cricket Score Prediction Analysis", fontsize=14, fontweight="bold")

        # chart 1 - scatter plot of runs vs predicted score with a regression line on top
        axes[0].scatter(runs_list, predicted_list, color="blue", label="Predicted Scores", zorder=5)
        axes[0].plot(sorted(runs_list), sorted(regression_line), color="red", label="Regression Line")
        axes[0].set_title("Runs vs Predicted Score")
        axes[0].set_xlabel("Runs Scored")
        axes[0].set_ylabel("Predicted Final Score")
        axes[0].legend()
        axes[0].grid(True)

        # chart 2 - the main over by over graph
        # blue solid line shows actual runs from over 1 to current over
        # orange dashed line continues from current over all the way to over 20 as prediction
        # both lines meet at the red dot which is where we are right now in the match
        axes[1].plot(past_overs, past_runs,
                     color="blue", marker="o", linewidth=2.5, markersize=5,
                     label="Actual Runs (Past)")

        axes[1].plot(future_overs, future_predicted,
                     color="orange", marker="s", linewidth=2.5, markersize=5,
                     linestyle="--", label="Predicted Score (Future)")

        # red dot marks the current position in the match
        axes[1].scatter([int(overs)], [int(runs)],
                        color="red", s=100, zorder=6,
                        label=f"Now: Over {int(overs)}, Runs {int(runs)}")

        # vertical dotted line to show where past ends and future begins
        axes[1].axvline(x=overs, color="gray", linestyle=":", linewidth=1.5)
        axes[1].text(overs + 0.2, min(past_runs) * 0.98,
                     "← Past | Future →", fontsize=8, color="gray")

        # light shading under both lines to make the graph easier to read
        axes[1].fill_between(past_overs,   past_runs,        alpha=0.12, color="blue")
        axes[1].fill_between(future_overs, future_predicted, alpha=0.12, color="orange")

        # showing the run value on top of each past dot
        for o, r in zip(past_overs, past_runs):
            axes[1].annotate(f"{r}", (o, r),
                             textcoords="offset points", xytext=(0, 8),
                             ha="center", fontsize=7, color="blue", fontweight="bold")

        # showing the predicted value below each future square
        for o, p in zip(future_overs[1:], future_predicted[1:]):
            axes[1].annotate(f"{p}", (o, p),
                             textcoords="offset points", xytext=(0, -14),
                             ha="center", fontsize=7, color="darkorange", fontweight="bold")

        axes[1].set_title(f"Over by Over — Entered at Over {int(overs)}, Runs {int(runs)}")
        axes[1].set_xlabel("Overs →")
        axes[1].set_ylabel("Runs ↑")
        axes[1].set_xticks(range(1, total_overs + 1))
        axes[1].set_xticklabels([str(i) for i in range(1, total_overs + 1)], fontsize=7)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, linestyle="--", alpha=0.4)

        # table at the bottom of chart 2 showing current over + key future overs
        # i only show every 5th over so the table doesn't get too crowded
        key_overs  = [int(overs)] + [o for o in range(int(overs)+1, total_overs+1) if o % 5 == 0 or o == 20]
        table_rows = []
        for o in key_overs:
            if o <= overs:
                r = int(run_rate * o)
                table_rows.append([f"Over {o}", str(r), "Actual"])
            else:
                extra = o - overs
                proj  = int(runs + adjusted_run_rate * extra)
                table_rows.append([f"Over {o}", str(proj), "Predicted"])

        tbl = axes[1].table(
            cellText  = table_rows,
            colLabels = ["Over", "Runs", "Type"],
            cellLoc   = "center",
            loc       = "bottom",
            bbox      = [0, -0.50, 1, 0.35]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        # making the header row green to match the app theme
        for j in range(3):
            tbl[0, j].set_facecolor("#4CAF50")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # chart 3 - scatter plot to see how run rate affects the predicted score
        axes[2].scatter(run_rate_list, predicted_list, color="purple", zorder=5)
        axes[2].set_title("Run Rate vs Predicted Score")
        axes[2].set_xlabel("Current Run Rate (runs/over)")
        axes[2].set_ylabel("Predicted Final Score")
        axes[2].grid(True)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)
        plt.show()

    def show_history():
        if not prediction_queue:
            messagebox.showinfo("History", "No predictions yet!")
            return
        # showing all predictions stored in the queue, max 5 at a time
        history_text = "Last 5 Predictions:\n" + "\n".join(
            [f"  Entry {i+1}: {score}" for i, score in enumerate(prediction_queue)]
        )
        messagebox.showinfo("Prediction History (Queue)", history_text)

    def clear():
        # just clearing all three input fields and the result label
        runs_entry.delete(0, tk.END)
        overs_entry.delete(0, tk.END)
        wickets_entry.delete(0, tk.END)
        result_label.config(text="")

    def logout():
        # clearing the logged in user info and going back to the login screen
        logged_in_user["email"] = None
        logged_in_user["name"]  = None
        window.destroy()
        show_login_window()

    btn_frame = tk.Frame(window)
    btn_frame.pack(pady=5)

    tk.Button(btn_frame, text="Predict Score", font=("Arial", 11),
              bg="#4CAF50", fg="white", width=14, command=predict).grid(row=0, column=0, padx=5, pady=5)
    tk.Button(btn_frame, text="Show Graph",    font=("Arial", 11),
              bg="#2196F3", fg="white", width=14, command=show_graph).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(btn_frame, text="View History",  font=("Arial", 11),
              bg="#FF9800", fg="white", width=14, command=show_history).grid(row=1, column=0, padx=5, pady=5)
    tk.Button(btn_frame, text="Clear",         font=("Arial", 11),
              bg="#f44336", fg="white", width=14, command=clear).grid(row=1, column=1, padx=5, pady=5)
    tk.Button(btn_frame, text="Logout",        font=("Arial", 11),
              bg="#607D8B", fg="white", width=14, command=logout).grid(row=2, column=0, columnspan=2, pady=5)

    window.mainloop()


def show_login_window():
    login_window = tk.Tk()
    login_window.title("Login - Cricket Score Predictor")
    login_window.geometry("350x220")
    login_window.resizable(False, False)

    tk.Label(login_window, text="Cricket Score Prediction",
             font=("Arial", 16, "bold")).pack(pady=20)
    tk.Label(login_window, text="Sign in with your Google account to continue",
             font=("Arial", 10), fg="gray").pack(pady=5)
    tk.Button(login_window, text="Sign in with Google",
              font=("Arial", 12, "bold"), bg="#4285F4", fg="white",
              padx=20, pady=8,
              command=lambda: start_google_login(login_window)).pack(pady=20)

    login_window.mainloop()


if __name__ == "__main__":
    show_login_window()
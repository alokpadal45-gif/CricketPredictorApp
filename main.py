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

load_dotenv()


CLIENT_ID    = os.environ.get("GOOGLE_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
REDIRECT_URI = "http://localhost:8080"
SCOPE        = "openid email profile"

logged_in_user  = {"email": None, "name": None}
match_history   = []
last_prediction = {}
prediction_queue = deque(maxlen=5)
CSV_FILE = "cricket_data.csv"


def setup_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Runs", "Overs", "Wickets", "PredictedScore"])


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


def save_to_csv(runs, overs, wickets, predicted):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([runs, overs, wickets, predicted])


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query  = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        if "code" in params:
            self.server.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Login successful! You can close this tab.</h2></body></html>")
        else:
            self.server.auth_code = None
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Login failed. Please try again.</h2></body></html>")

    def log_message(self, format, *args):
        pass


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


def get_user_info(access_token):
    url = "https://www.googleapis.com/oauth2/v2/userinfo"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {access_token}")
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def start_google_login(login_window):
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
        server = HTTPServer(("localhost", 8080), OAuthCallbackHandler)
        server.auth_code = None
        server.timeout   = 120
        webbrowser.open(auth_url)
        server.handle_request()
        auth_code = server.auth_code
        server.server_close()

        if not auth_code:
            login_window.after(0, lambda: messagebox.showerror(
                "Login Failed", "Google authentication failed or was cancelled."))
            return
        try:
            tokens      = exchange_code_for_tokens(auth_code)
            access_token = tokens["access_token"]
            user_info   = get_user_info(access_token)
            logged_in_user["email"] = user_info.get("email", "Unknown")
            logged_in_user["name"]  = user_info.get("name",  "User")
            login_window.after(0, lambda: open_cricket_app(login_window))
        except Exception as e:
            login_window.after(0, lambda: messagebox.showerror(
                "Login Error", f"Failed to complete login:\n{e}"))

    threading.Thread(target=oauth_flow, daemon=True).start()


# ── helper: predict score for any runs/overs/wickets value ──────────────────
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
    login_window.destroy()
    setup_csv()
    load_csv()

    window = tk.Tk()
    window.title("Cricket Score Prediction System")
    window.geometry("420x560")
    window.resizable(False, False)

    tk.Label(window,
             text=f"Welcome, {logged_in_user['name']}!  |  {logged_in_user['email']}",
             font=("Arial", 9, "italic"), fg="white", bg="#4CAF50",
             anchor="w", padx=10).pack(fill="x")

    tk.Label(window, text="Cricket Score Prediction System",
             font=("Arial", 14, "bold")).pack(pady=12)

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

    result_label = tk.Label(window, text="", font=("Arial", 11), justify="left")
    result_label.pack(pady=10)

    def predict():
        try:
            runs    = float(runs_entry.get())
            overs   = float(overs_entry.get())
            wickets = int(wickets_entry.get())
            total_overs = 20

            if overs == 0:
                result_label.config(text="Please enter overs played.", fg="red"); return
            if overs > total_overs:
                result_label.config(text="Overs cannot exceed 20.", fg="red"); return
            if wickets < 0 or wickets > 10:
                result_label.config(text="Wickets must be between 0 and 10.", fg="red"); return

            run_rate          = runs / overs
            remaining_overs   = total_overs - overs
            adjusted_run_rate = run_rate * (1 - (wickets * 0.03))
            if adjusted_run_rate < 0:
                adjusted_run_rate = 0
            basic_predicted = runs + (adjusted_run_rate * remaining_overs)

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

            final_predicted = max(final_predicted, runs)

            save_to_csv(runs, overs, wickets, int(final_predicted))
            match_history.append({"runs": runs, "overs": overs,
                                   "wickets": wickets, "predicted": int(final_predicted)})

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

        runs      = last_prediction["runs"]
        overs     = last_prediction["overs"]
        wickets   = last_prediction["wickets"]
        total_overs = 20

        # ── build over by over data from ONE prediction ──────────────────────
        # PAST  : over 1 to current over  → calculate how many runs each over
        #         assuming runs came in evenly (runs/overs per over)
        # FUTURE: current over+1 to 20    → predict score for each future over
        #         using same run rate adjusted for wickets

        run_rate          = runs / overs
        adjusted_run_rate = run_rate * (1 - (wickets * 0.03))
        if adjusted_run_rate < 0:
            adjusted_run_rate = 0

        past_overs      = []   # x values for past  (1 to current over)
        past_runs       = []   # actual cumulative runs at each past over

        future_overs    = []   # x values for future (current over to 20)
        future_predicted = []  # predicted cumulative score at each future over

        # past: estimate cumulative runs at each over up to now
        for o in range(1, int(overs) + 1):
            past_overs.append(o)
            past_runs.append(int(run_rate * o))   # even spread of runs

        # future: start from current runs, add adjusted run rate each over
        for o in range(int(overs), total_overs + 1):
            future_overs.append(o)
            extra_overs      = o - overs
            projected_runs   = runs + (adjusted_run_rate * extra_overs)
            future_predicted.append(int(projected_runs))

        # ── draw the 3 charts ────────────────────────────────────────────────
        runs_list      = [m["runs"]             for m in match_history]
        predicted_list = [m["predicted"]        for m in match_history]
        run_rate_list  = [m["runs"] / m["overs"] for m in match_history]

        X_reg = np.array(runs_list).reshape(-1, 1)
        y_reg = np.array(predicted_list)
        reg_model = LinearRegression()
        reg_model.fit(X_reg, y_reg)
        regression_line = reg_model.predict(X_reg)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Cricket Score Prediction Analysis", fontsize=14, fontweight="bold")

        # ── CHART 1 ── runs vs predicted with regression line
        axes[0].scatter(runs_list, predicted_list, color="blue", label="Predicted Scores", zorder=5)
        axes[0].plot(sorted(runs_list), sorted(regression_line), color="red", label="Regression Line")
        axes[0].set_title("Runs vs Predicted Score")
        axes[0].set_xlabel("Runs Scored")
        axes[0].set_ylabel("Predicted Final Score")
        axes[0].legend()
        axes[0].grid(True)

        # ── CHART 2 ── over by over: past actual + future predicted
        # BLUE SOLID LINE   = past overs (actual runs already scored, going up)
        # ORANGE DASHED LINE = future overs (predicted score, continuing upward)
        # RED DOT           = current position (where we are now)
        # Both lines connect at current over — exactly like sir's drawing

        axes[1].plot(past_overs, past_runs,
                     color="blue", marker="o", linewidth=2.5, markersize=5,
                     label="Actual Runs (Past)")

        axes[1].plot(future_overs, future_predicted,
                     color="orange", marker="s", linewidth=2.5, markersize=5,
                     linestyle="--", label="Predicted Score (Future)")

        # red dot at current over showing exact position
        axes[1].scatter([int(overs)], [int(runs)],
                        color="red", s=100, zorder=6,
                        label=f"Now: Over {int(overs)}, Runs {int(runs)}")

        # vertical dashed line separating past from future
        axes[1].axvline(x=overs, color="gray", linestyle=":", linewidth=1.5)
        axes[1].text(overs + 0.2, min(past_runs) * 0.98,
                     "← Past | Future →", fontsize=8, color="gray")

        # shade past region light blue, future region light orange
        axes[1].fill_between(past_overs,   past_runs,       alpha=0.12, color="blue")
        axes[1].fill_between(future_overs, future_predicted, alpha=0.12, color="orange")

        # label run values on past dots
        for o, r in zip(past_overs, past_runs):
            axes[1].annotate(f"{r}", (o, r),
                             textcoords="offset points", xytext=(0, 8),
                             ha="center", fontsize=7, color="blue", fontweight="bold")

        # label predicted values on future squares
        for o, p in zip(future_overs[1:], future_predicted[1:]):
            axes[1].annotate(f"{p}", (o, p),
                             textcoords="offset points", xytext=(0, -14),
                             ha="center", fontsize=7, color="darkorange", fontweight="bold")

        axes[1].set_title(f"Over by Over — Entered at Over {int(overs)}, Runs {int(runs)}")
        axes[1].set_xlabel("Overs →")
        axes[1].set_ylabel("Runs ↑")
        axes[1].set_xticks(range(1, total_overs + 1))
        axes[1].set_xticklabels([str(i) for i in range(1, total_overs + 1)],
                                 fontsize=7)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, linestyle="--", alpha=0.4)

        # small table below chart 2 showing key overs
        # shows: current over + every 5th future over
        key_overs = [int(overs)] + [o for o in range(int(overs)+1, total_overs+1) if o % 5 == 0 or o == 20]
        table_rows = []
        for o in key_overs:
            if o <= overs:
                r = int(run_rate * o)
                table_rows.append([f"Over {o}", str(r), "Actual"])
            else:
                extra   = o - overs
                proj    = int(runs + adjusted_run_rate * extra)
                table_rows.append([f"Over {o}", str(proj), "Predicted"])

        tbl = axes[1].table(
            cellText   = table_rows,
            colLabels  = ["Over", "Runs", "Type"],
            cellLoc    = "center",
            loc        = "bottom",
            bbox       = [0, -0.50, 1, 0.35]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for j in range(3):
            tbl[0, j].set_facecolor("#4CAF50")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # ── CHART 3 ── run rate vs predicted score
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
        history_text = "Last 5 Predictions:\n" + "\n".join(
            [f"  Entry {i+1}: {score}" for i, score in enumerate(prediction_queue)]
        )
        messagebox.showinfo("Prediction History (Queue)", history_text)

    def clear():
        runs_entry.delete(0, tk.END)
        overs_entry.delete(0, tk.END)
        wickets_entry.delete(0, tk.END)
        result_label.config(text="")

    def logout():
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
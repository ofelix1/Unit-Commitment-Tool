import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from config import location1


# Data Loaded
df = pd.read_excel(location1) 
df['Date'] = pd.to_datetime(df['Date'])

# Outliers removed using the Interquartile Range (IQR) Method
def remove_outliers_iqr(data, column):
   """ Removes outliers using the IQR method for a given column. """
   Q1 = data[column].quantile(0.25)
   Q3 = data[column].quantile(0.75)
   IQR = Q3 - Q1
   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR
   return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

df = remove_outliers_iqr(df, 'Load_Served')
df = remove_outliers_iqr(df, 'Temperature')


# Polynomial regression for Load_Served vs Temperature
X_load = df[['Temperature']]
y_load = df['Load_Served']
load_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
load_model.fit(X_load, y_load)

predicted_y = load_model.predict(X_load)
residuals = np.abs(y_load - predicted_y)
threshold = 1.5 * np.std(residuals)  
# Threshold limits
df_filtered = df[residuals <= threshold]

# Train models for dependent variables
dependent_variables = [
   'Gen_Unit_1', 'Gen_Unit_2', 'Gen_Unit_3',
   'Gen_Unit_4', 'Gen_Unit_5', 'Gen_Unit_6',
   'Gen_Unit_7', 'MISO Import/Export'
]
models = {}
for var in dependent_variables:
   df = remove_outliers_iqr(df, var) 
   model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
   model.fit(df[['Temperature', 'Load_Served']], df[var])
   models[var] = model

# Min/max for resource outputs
output_limits = {
   'Gen_Unit_1': (0.0, 900), 'Gen_Unit_2': (0.0, 250),
   'Gen_Unit_3': (0.0, 250), 'Gen_Unit_4': (0.0, 202),
   'Gen_Unit_5': (0.0, 368), 'Gen_Unit_6': (0.0, 495),
   'Gen_Unit_7': (0.0, 437)
}
# GUI 
root = tk.Tk()
root.title("Power Plant Load Prediction")

# Temp input 
tk.Label(root, text="Temperature (°F):", font=("Arial", 14)).pack()
temp_entry = tk.Entry(root, font=("Arial", 14))
temp_entry.pack()

# Predict Button
predict_button = tk.Button(root, text="Predict", font=("Arial", 12), command=lambda: display_results(float(temp_entry.get())))
predict_button.pack()

# Labels for results
result_labels = {}
for var in ['Load_Served'] + dependent_variables:
   label = tk.Label(root, text=f"{var}: ", font=("Arial", 12))
   label.pack()
   result_labels[var] = label

# Label for forecast error (MAPE)
error_label = tk.Label(root, text="Forecast Error (MAPE): --%", font=("Arial", 12, "bold"), fg="red")
error_label.pack()

# Plot before update 
fig, ax = plt.subplots(figsize=(10, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
def display_results(temperature):
   """ Predict values and update GUI & chart dynamically. """
   load_served = load_model.predict(pd.DataFrame([[temperature]], columns=['Temperature']))[0]

   # Predict other values based on Load_Served
   predictions = {'Load_Served': load_served}
   for var, model in models.items():
       predicted_value = model.predict(pd.DataFrame([[temperature, load_served]],
                                                    columns=['Temperature', 'Load_Served']))[0]
       if var in output_limits:
           min_val, max_val = output_limits[var]
           predicted_value = min(max(predicted_value, min_val), max_val)
       predictions[var] = predicted_value


   # Update Labels
   for var, value in predictions.items():
       if var == "MISO Import/Export":
           result_labels[var].config(text=f"{var}: {value:.2f} (Imports (-) / Exports (+))")
       else:
           result_labels[var].config(text=f"{var}: {value:.2f}")
   # Forecast error (MAPE) calculation
   nearest_data = df.iloc[(df['Temperature'] - temperature).abs().argsort()[:5]]
   actual_values = nearest_data['Load_Served'].values
   predicted_values = load_model.predict(nearest_data[['Temperature']])
   if len(actual_values) > 0:
       mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
       error_label.config(text=f"Forecast Error (MAPE): {mape:.2f}%")
   else:
       error_label.config(text="Forecast Error (MAPE): No Data")
 
   ax.clear()

   # Replot filters data
   ax.scatter(df_filtered['Temperature'], df_filtered['Load_Served'],
              s=5, alpha=0.5, label="Filtered Data (Close to Fit)")
   
   # Recalculate and Replot Polynomial Fit
   temp_range = np.linspace(df['Temperature'].min(), df['Temperature'].max(), 100)
   predicted_curve = load_model.predict(pd.DataFrame(temp_range, columns=['Temperature']))
   ax.plot(temp_range, predicted_curve, color='r', linewidth=2, label="Polynomial Fit")

   # X-axis limits
   ax.set_xlim(0, 100)
   ax.xaxis.set_major_locator(MultipleLocator(10))
   
   # Labels/legend
   ax.set_title('Load Served vs Temperature', fontsize=12)
   ax.set_xlabel('Temperature (°F)', fontsize=10)
   ax.set_ylabel('Load Served', fontsize=10)
   ax.legend()
   ax.grid(True)
   canvas.draw()


root.mainloop()
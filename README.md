#  Unit Commitment Tool

**1. Forcasted Data**
   -A polynomial regression model is used to correlate the relationships between temperature, load, and unit MW committed.
   -Data outliers are filtered through the Interquartile Range (IQR) method. Values falling outside the 75th to 25th percentile range are considered outliers and removed.
   -The Mean Absolute Percentage Error (MAPE) is used to evaluate the performance of the model, with a lower MAPE indicating better accuracy in the model's predictions.

 **2. Use Case**
    -This is a replica of a project I worked on at my current role as Power Market Analyst, where the goal was to predict the possibility of units being committed, along with the  
     estimated load obligation and MW commitment based on the past correlation between temperature and load.
    -A randomized data set was used for this model. 
    
**3. How it Works**
   -Input temperature and GUI outputs forcasted data.(MW unit Commitment, MISO load obligation, import export, and percentage error)
  
<img width="500" alt="Load Prediciton Dash" src="https://github.com/user-attachments/assets/1e8b7e1e-342d-4b7e-83f1-5722172713dc" />

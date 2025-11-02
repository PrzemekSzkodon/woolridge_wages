import polars as pl
import pandas as pd
import pyreadr
import requests

"""
1. Regression modeling and interpretation of results. Jacob Mincer (1958) “Investment in
Human Capital and Personal Income Distribution”, JPE 66, 281–302 spells a model of education
as an investment, where human capital H equals

H= Aeβeduc
,
where educ is educational attainment, β is the return to an investment of the annual earnings
amount (that gives you an extra year of education, and so increases human capital), and A
includes all other factors influencing human capital. If one unit of H can be rented by a firm
at price P, a person with capital H gets wage= H×P:

log (wage) = log P + βeduc + log A.

Using the data on workers’ wages, education, experience, etc. from wage1.csv or wage1.dta
(taken from the 1976 Current Population Survey), run a regression of the logarithm of wage on
the education and tenure. Interpret and critically assess your results. Pay attention to possible
biases.
"""

def download_wage1_data():
    wage1_url = "http://fmwww.bc.edu/ec-p/data/wooldridge/wage1.dta"
    response = requests.get(wage1_url) #this gets the file from the url
    open('wage1.dta', 'wb').write(response.content) #this writes the file to the current directory, wage1.dta is the file name, wb is write and binary, response.content is the content of the file
    df_wage1 = pd.read_stata('wage1.dta')
    return df_wage1

def load_wage1_data():
    df_wage1 = pd.read_stata('wage1.dta')
    return df_wage1 #simple function to load the data




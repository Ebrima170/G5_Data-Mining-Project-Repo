#%%
import pandas as pd
# %%
CO2 = pd.read_csv("co2_pcap_cons.csv")
gdp= pd.read_csv("gdp_pcap.csv")
lex = pd.read_csv("lex.csv")
pop = pd.read_csv("pop.csv")
print(CO2.head())
print(gdp.head())
print(lex.head())
print(pop.head())
# %%
CO2_long = CO2.melt(
    id_vars=['geo', 'name'],   # Columns to keep
    var_name='year',           # The melted column name
    value_name='co2'           # The numeric values column name
)
print(CO2_long.head())
CO2_long.to_csv("co2_long.csv", index=False)
# %%
gdp_long = gdp.melt(
    id_vars=['geo', 'name'],
    var_name='year',
    value_name='gdp'
)
print(gdp_long.head())
gdp_long.to_csv("gdp_long.csv", index=False)
# %%
lex_long = lex.melt(
    id_vars=['geo', 'name'],
    var_name='year',
    value_name='lex'
)
print(lex_long.head())
lex_long.to_csv("lex_long.csv", index=False)
# %%
pop_long = pop.melt(
    id_vars=['geo', 'name'],
    var_name='year',
    value_name='pop'
)
print(pop_long.head())
pop_long.to_csv("pop_long.csv", index=False)
# %%

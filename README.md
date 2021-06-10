# BNT162b2 mRNA vaccinations in Israel: understanding the impact and improving the vaccination policies by redefining the immunized population


1. covid-19 data + cbs data on Israel can be found in:
  - israel_data/*.csv
  
2. code for running analysis:
  - israel_data/analyze_vacc_data_per_city.ipynb
  - israel_data/load_israeli_data_original.ipynb
  
3. seirsplus simulation files can be found in:
<img src="https://github.com/CoronaPolicy/prediction_model/blob/main/seirsplus_quarentine.png" width="400">
<img align="right" src="https://github.com/ryansmcgee/seirsplus/blob/master/images/network_p.png" height="220">

  - seir_plus_changes/*.py
  - these files can be used for running the simulation in : https://github.com/ryansmcgee/seirsplus with our corrections for Israel and for vaccination policies
  
4. code for running simulation on multiple seeds and vaccination policies:
  - Code_seir/simulation_impact_factor_multiple_runs.py
  - Code_seir/vaccinations/analyze_graph_and_results_vaccinations_all_seeds.ipynb
  
5. Python Requirments:
  -python >= 3.7
  -matplotlib
  -scipy
  -numpy
  -pandas
  -networkx
  -farz

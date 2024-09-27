import subprocess

print("Running Experiments.")
file_list = ['SyntheticDatasets/linear/linear_x.py',\
             'SyntheticDatasets/linear/linear_y.py', \
             'SyntheticDatasets/nonlinear/nonlinear_x.py',\
             'SyntheticDatasets/nonlinear/nonlinear_y.py',\
             'Communities and Crime/CommunitiesAndCrimeSimulation_01.py',\
             'Communities and Crime/CommunitiesAndCrimeSimulation_02.py',\
             'Company Bankruptcy Prediction/CompanyBankruptcy.py']

for file in file_list:
    subprocess.run(['python', file])

print("Running Parameter Sensitivity Analysis.")
par_file_list =['SyntheticDatasets/linear/par_sen.py',\
                'SyntheticDatasets/nonlinear/par_sen.py',\
                'Communities and Crime/par_sen.py',\
                'Company Bankruptcy Prediction/par_sen.py']


for file in par_file_list:
    subprocess.run(['python', file])

print("All files have been executed in sequence. The result can be found in the file <dataset/result>")

import subprocess

file_list = ['par_sen_1_method1_alpha.py', 'par_sen_1_method2_alpha.py', 'par_sen_1_method2_beta.py',\
             'par_sen_2_method1_alpha.py', 'par_sen_2_method2_alpha.py', 'par_sen_2_method2_beta.py']

for file in file_list:
    subprocess.run(['python', file])

print("All files have been executed in sequence.")

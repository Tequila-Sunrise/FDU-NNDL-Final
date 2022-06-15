import os

dir_name = 'Oracle/oracle_fs/seq/oracle_200_5_shot'

with open('./npz_sum.txt', 'w') as f:
    for num in range(200):
        f.writelines(f'{dir_name}/{num}.npz\n')

import random
import numpy as np
from case_config import *
from constraints import check_constraints

def generate_random_individual():
    ind = []
    # 连续几何变量
    for r in VAR_RANGES_GEO:
        ind.append(random.uniform(r[0], r[1]))
    
    # 离散变量 (映射为连续索引)
    ind.append(random.uniform(0, len(VAL_FC)-0.01)) # fc
    ind.append(random.uniform(0, len(VAL_FY)-0.01)) # fy
    
    # 材料连续变量
    ind.append(random.uniform(*VAR_RANGES_MAT[0])) # dr
    ind.append(random.uniform(*VAR_RANGES_MAT[1])) # dp
    
    # 离散预应力
    ind.append(random.uniform(0, len(VAL_NPB)-0.01)) # npb
    ind.append(random.uniform(0, len(VAL_NPW)-0.01)) # npw
    
    # 预应力连续
    ind.append(random.uniform(*VAR_RANGES_MAT[2])) # sigma
    ind.append(random.uniform(*VAR_RANGES_MAT[3])) # hp_ratio
    
    return ind

def main():
    N_SAMPLES = 100
    valid_count = 0
    
    print(f"Checking {N_SAMPLES} random samples...")
    
    for i in range(N_SAMPLES):
        ind = generate_random_individual()
        # Only print details for first few failures or any success
        debug_flag = (i < 5) 
        penalty = check_constraints(ind, debug=debug_flag)
        
        if penalty == 0:
            valid_count += 1
            print(f"Sample {i}: VALID")
        elif debug_flag:
            print(f"Sample {i}: INVALID")
            
    print(f"\nSummary: {valid_count}/{N_SAMPLES} valid solutions found.")

if __name__ == "__main__":
    main()
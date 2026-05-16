import pandas as pd
import cvxpy as cp
import numpy as np
import datetime

from tqdm import tqdm

#----------------------------------------------
# Nacitanie a spracovanie dat
#----------------------------------------------
def nacitanie_dat(subor):
    j=1
    values = []
    p=0
    df = pd.read_excel(subor)                      
    r = len(df.iloc[:,1]) - 2  # kvoli poslednym dvom riadkom aby mi optimalizacia zbehla 
    for i in tqdm(range(0,r)):
        if i > 14:
            udaje = df.iloc[i-15:i,1].tolist()
            p = ((i+2) % 15) -3
            perioda =  p 
            j = (i+2) // 15 
    
            odchylka = df.iloc[j,2]
            value = [perioda, odchylka] + udaje
            values.append(value) 
    stlpec = ["perioda"] + ["odchylka"] + [f"data_{k+1}" for k in range(15)]
    zoznam = pd.DataFrame(data = values, columns=stlpec)
    #zoznam.to_excel("...\\data\\Dataset_trenovaci.xlsx",index=False)   
    print(zoznam)
    
    
    return zoznam 

#-------------------------------------
# Vytvorenie A, b
#-------------------------------------
def vytvor_datasety(zoznam, perioda):
    per = zoznam[zoznam["perioda"] == perioda]
    b = per["odchylka"].values * 4 # *4 aby sme mali [MW]
    A = per.iloc[:,2:17].values
    return A, b




#====================
# Povodny dataset (P)
#====================


#----------------------------------------
# Optimalizacia (P)
#----------------------------------------
def vytvor_model(A, b, f=cp.sum_squares):
  x = cp.Variable(A.shape[1])
  n = A.shape[0]
  w = np.ones(n)
  r = (A @ x - b)
  f_r = f(r)

  if f_r.is_scalar():
      objective = cp.Minimize(f_r)
  else:
      objective = cp.Minimize(w @ f_r)
  prob = cp.Problem(objective)
  return prob, x 



def logisticka_regresia(A, b, lam = 0.0, verbose = True):
    y = (b>0).astype(float) # y={0,1}
    x = cp.Variable(A.shape[1])
    
    z = A @ x
    loss = cp.sum(cp.logistic(z) - cp.multiply(y, z))
    objective = cp.Minimize(loss + lam * cp.norm(x, 1))
    
    problem = cp.Problem(objective)
    problem.solve(solver=cp.ECOS, verbose=verbose)  
#    problem.solve(solver=cp.CLARABEL, verbose=verbose,
#                  reduced_tol_gap_abs=10000.0,
#                  reduced_tol_gap_rel=10.0,
#                  reduced_tol_feas=20.0) #, reduced_tol_gap_abs=1e-2)
    return x.value

def optimalizuj(A, b, f=cp.sum_squares,
                verbose=False):
    prob, x = vytvor_model(A, b, f)
    prob.solve(solver=cp.CLARABEL, verbose=verbose,
                  reduced_tol_gap_abs=10000.0,
                  reduced_tol_gap_rel=10.0,
                  reduced_tol_feas=20.0)
    return x.value




#----------------------------------------
# Optimalne vahy (P)
#----------------------------------------

def optimalne_v(zoznam):
    periody = zoznam["perioda"].unique()
    values_3 = []
    
    for per in tqdm(periody):
        perioda = per
        A ,b = vytvor_datasety(zoznam, perioda)

       # Normy skalar
        norma_2 = optimalizuj(A, b, f = lambda r: cp.norm(r,2))
        norma_1 = optimalizuj(A, b, f = lambda r: cp.norm(r,1))  
        
        norma_max = optimalizuj(A, b, f = lambda r: cp.norm(r,'inf'))
        
        # F vektor
        penal = optimalizuj(A, b, f = lambda r: cp.sum(cp.abs(r) ** 1.5))
        huber = optimalizuj(A, b, f = lambda r: cp.sum(cp.huber(r,50)))
        
        # logisticka regresia
        log_reg = logisticka_regresia(A, b, lam = 0.0, verbose=True)
        value = [perioda, norma_2, norma_1, norma_max, penal, huber, log_reg]
        values_3.append(value)
        
    stplec_3 = ["perioda","norma_2","norma_1","norma_max", "penal","huber","log_reg"]
    opt_vahy_x = pd.DataFrame(data = values_3, columns=stplec_3)
    print("Optimalne vahy kazdej metody: ")
    print(opt_vahy_x)
    opt_vahy_x.to_excel("...\\data\\optimalne_vahy_P.xlsx",sheet_name= "Sheet1",index=True) 
    
    return opt_vahy_x


#--------------------------------------
# Predikcia (P)
#--------------------------------------
def sprav_excel_tabulku(zoznam_2, opt_vahy_x):
    probl = []
    row_w = []
    start = datetime.datetime(2025, 9, 1, 0, 17)
    for i in tqdm(range(len(zoznam_2))):
      current = start + datetime.timedelta(minutes=i)
      cas = current.strftime("%d.%m %H:%M")

      odchylka = (zoznam_2.iloc[i,1]) * 4 # *4 aby sme mali [MW]
      odchylka_r = zoznam_2.iloc[i,1]
      perioda = zoznam_2.iloc[i,0]     
 
      
      # Ziskava hodnoty
      A_row = zoznam_2.iloc[i,2:17].tolist()
      row_w = opt_vahy_x.loc[opt_vahy_x["perioda"] == perioda].iloc[0]

      predikcie = [
          np.dot(A_row, row_w["norma_2"]),
          np.dot(A_row, row_w["norma_1"]),
          np.dot(A_row, row_w["norma_max"]),
          np.dot(A_row, row_w["penal"]),
          np.dot(A_row, row_w["huber"]),
          np.dot(A_row, row_w["log_reg"])
          ]  
      value = [cas, perioda] + A_row + predikcie + [odchylka, odchylka_r]
      probl.append(value)
      #
    stlp = ["cas","model"] + [f"data_{k+1}" for k in range(15)] + \
        ["dvojkova norma","jednotkova norma","maximova norma","penalizacna funkcia",
         "hubertova funkcia"," logisticka regresia", "realne odchylka [MW]","realne odchylka [MWh]"]
    excel = pd.DataFrame(data = probl, columns=stlp)
    print("Predikcie odchylky nasimi metodami")
    print(excel)
    excel.to_excel("...\\data\\test_P.xlsx",sheet_name= "Sheet1",index=True)  
    
    return excel

#--------------------------------   
# Metriky modelov (P)
#--------------------------------
def vyhodnotenie(zoznam,opt_vahy_x):
    modely = ["norma_2", "norma_1", "norma_max", "penal", "huber","log_reg"]
    vysledky = {} 
    periody = zoznam["perioda"].unique()
    

    
    for perioda in tqdm(periody):
        per = zoznam[zoznam["perioda"] == perioda]
        row = opt_vahy_x.loc[opt_vahy_x["perioda"] == perioda].iloc[0]
        b = (per["odchylka"].values)[:-3] * 4  #posledne 3 su Nan  *4 aby sme mali [MW]
        A = per.iloc[:-3, 2:17].values  
        
        perioda_metrics = {}
        
        for model in modely:
            vaha_modelu = row[model]
            
            if model == "log_reg":
                scores = A @ vaha_modelu
                

                y_pred = (scores > 0).astype(int)
                y_true = (b > 0).astype(int)
                

                accuracy = np.mean(y_pred == y_true)
                

                tp = np.sum((y_pred == 1) & (y_true == 1))  # True Positive
                tn = np.sum((y_pred == 0) & (y_true == 0))  # True Negative
                fp = np.sum((y_pred == 1) & (y_true == 0))  # False Positive
                fn = np.sum((y_pred == 0) & (y_true == 1))  # False Negative
                
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                

                perioda_metrics[f"{model}_accuracy"] = accuracy
                perioda_metrics[f"{model}_precision"] = precision
                perioda_metrics[f"{model}_recall"] = recall
                perioda_metrics[f"{model}_f1"] = f1
    
                # skutocne zaporne, predikcia kladna
                fn_znam = np.sum((b < 0) & (y_pred > 0))
                # skutocne kladne, predikcia zaporna  
                fp_znam = np.sum((b > 0) & (y_pred < 0))
                celkom  = len(b)
                
                print(f"Perioda {perioda:3d} | {model:10s} | "
                      f"zap-klad: {fn_znam:3d} ({100*fn_znam/celkom:.1f}%) | "
                      f"klad-zap: {fp_znam:3d} ({100*fp_znam/celkom:.1f}%)")
            else:
                r = A @ vaha_modelu - b
                
    
                mse = np.mean(r**2)
                wape = np.sum(np.abs(r)) / np.sum(np.abs(b) + 1e-12)
                ss_res = np.sum(r**2)
                ss_tot = np.sum((b - np.mean(b))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                
    
                perioda_metrics[f"{model}_Mse"] = mse
                perioda_metrics[f"{model}_Wape"] = wape
                perioda_metrics[f"{model}_R2"] = r2
            

        vysledky[perioda] = perioda_metrics
        
    df_vysledky = pd.DataFrame.from_dict(vysledky , orient='index') 
    with pd.ExcelWriter("...\\data\\test_P.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:
         df_vysledky.to_excel(writer, sheet_name="Sheet2", index=True)
         
    print("metriky nasich metod: ")
    print(df_vysledky)
    
    return df_vysledky


#==================================================================
# Transformacia pomocou rozsirenia datasetu o absolutne hodnoty (A)
#==================================================================

def A_vytvor_model(A, b, f=cp.sum_squares):
  x = cp.Variable(A.shape[1])
  n = A.shape[0]
  w = np.ones(n)
  r = (A @ x - b)
  f_r = f(r)

  if f_r.is_scalar():
      objective = cp.Minimize(f_r)
  else:
      objective = cp.Minimize(w @ f_r)
  prob = cp.Problem(objective)
  return prob, x 


def A_logisticka_regresia(A, b, lam = 0.0, verbose = True):
    y = (b>0).astype(float) #  y={0,1}
    x = cp.Variable(A.shape[1])
    
    z = A @ x
    loss = cp.sum(cp.logistic(z) - cp.multiply(y, z))
    objective = cp.Minimize(loss + lam * cp.norm(x, 1))
    
    problem = cp.Problem(objective)
    problem.solve(solver=cp.ECOS, verbose=verbose)  
#    problem.solve(solver=cp.CLARABEL, verbose=verbose,
#                  reduced_tol_gap_abs=10000.0,
#                  reduced_tol_gap_rel=10.0,
#                  reduced_tol_feas=20.0) #, reduced_tol_gap_abs=1e-2)
    return x.value

def A_optimalizuj(A, b, f=cp.sum_squares,
                verbose=False):
    prob, x = A_vytvor_model(A, b, f)
    prob.solve(solver=cp.CLARABEL, verbose=verbose,
                  reduced_tol_gap_abs=10000.0,
                  reduced_tol_gap_rel=10.0,
                  reduced_tol_feas=20.0)
    return x.value

#----------------------------------------
# Optimalne vahy pre (A)
#----------------------------------------

def A_optimalne_v(zoznam):
    periody = zoznam["perioda"].unique()
    values_A = []
    
    for per in tqdm(periody):
        perioda = per
        A ,b = vytvor_datasety(zoznam, perioda)
        A = np.hstack((A, np.abs(A)))  
        
       # Normy skalar
        norma_2 = A_optimalizuj(A, b, f = lambda r: cp.norm(r,2))
        norma_1 = A_optimalizuj(A, b, f = lambda r: cp.norm(r,1))  
        
        norma_max = A_optimalizuj(A, b, f = lambda r: cp.norm(r,'inf'))
        penal = A_optimalizuj(A, b, f = lambda r: cp.sum(cp.abs(r) ** 1.5))
        huber = A_optimalizuj(A, b, f = lambda r: cp.sum(cp.huber(r,50))) 
        
        #Llogisticka regresia
        log_reg = logisticka_regresia(A, b, lam = 0.0, verbose=True)
        value = [perioda, norma_2, norma_1, norma_max, penal, huber, log_reg]
        values_A.append(value)
        
    stplec_A = ["perioda","norma_2","norma_1","norma_max", "penal","huber","log_reg"]
    opt_vahy_x_A = pd.DataFrame(data = values_A, columns=stplec_A)
    print("Optimalne vahy kazdej metody: ")
    print(opt_vahy_x_A)
    # Pre normalnu optimalizaciu
    opt_vahy_x_A.to_excel("...\\data\\optimalne_vahy_A.xlsx",sheet_name= "Sheet1",index=True)   
    return opt_vahy_x_A


#--------------------------------------
# Predikcia (A)
#--------------------------------------
def A_sprav_excel_tabulku(zoznam_2, opt_vahy_x_A):
    probl = []
    row_w = []
    start = datetime.datetime(2025, 9, 1, 0, 17)
    for i in tqdm(range(len(zoznam_2))):
      current = start + datetime.timedelta(minutes=i)
      cas = current.strftime("%d.%m %H:%M")

      odchylka = (zoznam_2.iloc[i,1]) * 4  # *4 aby sme mali [MW]
      odchylka_r = zoznam_2.iloc[i,1]
      perioda = zoznam_2.iloc[i,0]     
 
      
      #ziskava hodnoty
      A = zoznam_2.iloc[i,2:17].tolist()
      A_row = np.hstack((A, np.abs(A))) # rozsirenie o absolutne hodnoty
      row_w = opt_vahy_x_A.loc[opt_vahy_x_A["perioda"] == perioda].iloc[0]

      predikcie = [
          np.dot(A_row, row_w["norma_2"]),
          np.dot(A_row, row_w["norma_1"]),
          np.dot(A_row, row_w["norma_max"]),
          np.dot(A_row, row_w["penal"]),
          np.dot(A_row, row_w["huber"]),
          np.dot(A_row, row_w["log_reg"])
          ]  
      
      value = [cas, perioda] + A_row.tolist() + predikcie + [odchylka, odchylka_r]
      probl.append(value)
      
      
    stlp = ["cas","model"] + [f"data_{k+1}" for k in range(15)] + [f"abs_data_{k+1}" for k in range(15)] + \
        ["dvojkova norma","jednotkova norma","maximova norma","penalizacna funkcia",
         "hubertova funkcia"," logisticka regresia", "realne odchylka [MW]","realne odchylka [MWh]"]
          
    excel = pd.DataFrame(data = probl, columns=stlp)
    print("Predikcie odchylky nasimi metodami")
    print(excel)
    excel.to_excel("...\\data\\test_A.xlsx",sheet_name= "Sheet1",index=True)   
    return excel

#--------------------------------   
# Metriky modelov (A)
#--------------------------------
def A_vyhodnotenie(zoznam,opt_vahy_x_A):
    modely = ["norma_2", "norma_1", "norma_max", "penal", "huber","log_reg"]
    vysledky = {} 
    periody = zoznam["perioda"].unique()
    

    
    for perioda in tqdm(periody):
        per = zoznam[zoznam["perioda"] == perioda]
        row = opt_vahy_x_A.loc[opt_vahy_x_A["perioda"] == perioda].iloc[0]
        b = (per["odchylka"].values)[:-3] * 4  #posledne 3 su Nan  *4 aby sme mali [MW]
        A = per.iloc[:-3, 2:17].values  
        
        A = np.hstack((A, np.abs(A)))
        perioda_metrics = {}
        
        for model in modely:
            vaha_modelu = row[model]
            
            if model == "log_reg":
                scores = A @ vaha_modelu
                

                y_pred = (scores > 0).astype(int)
                y_true = (b > 0).astype(int)
                

                accuracy = np.mean(y_pred == y_true)
                

                tp = np.sum((y_pred == 1) & (y_true == 1))  # True Positive
                tn = np.sum((y_pred == 0) & (y_true == 0))  # True Negative
                fp = np.sum((y_pred == 1) & (y_true == 0))  # False Positive
                fn = np.sum((y_pred == 0) & (y_true == 1))  # False Negative
                
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                

                perioda_metrics[f"{model}_accuracy"] = accuracy
                perioda_metrics[f"{model}_precision"] = precision
                perioda_metrics[f"{model}_recall"] = recall
                perioda_metrics[f"{model}_f1"] = f1\
                    
                # skutocne zaporne, predikcia kladna
                fn_znam = np.sum((b < 0) & (y_pred > 0))
                # skutocne kladne, predikcia zaporna  
                fp_znam = np.sum((b > 0) & (y_pred < 0))
                celkom  = len(b)
                
                print(f"Perioda {perioda:3d} | {model:10s} | "
                      f"zap-klad: {fn_znam:3d} ({100*fn_znam/celkom:.1f}%) | "
                      f"klad-zap: {fp_znam:3d} ({100*fp_znam/celkom:.1f}%)")
                
            else:
                r = A @ vaha_modelu - b
                
    
                mse = np.mean(r**2)
                wape = np.sum(np.abs(r)) / np.sum(np.abs(b) + 1e-12)
                ss_res = np.sum(r**2)
                ss_tot = np.sum((b - np.mean(b))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                
    
                perioda_metrics[f"{model}_Mse"] = mse
                perioda_metrics[f"{model}_Wape"] = wape
                perioda_metrics[f"{model}_R2"] = r2
                

            

        vysledky[perioda] = perioda_metrics
        
    df_vysledky = pd.DataFrame.from_dict(vysledky , orient='index') 
    with pd.ExcelWriter("...\\data\\test_A.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:
         df_vysledky.to_excel(writer, sheet_name="Sheet2", index=True)
    print("metriky nasich metod: ")
    print(df_vysledky)
    
    return df_vysledky


#===========================================================
# Transformacia pomocou inverzneho hyperbolickeho sinusu (S)
#===========================================================
def S_vytvor_model(A, b, f=cp.sum_squares, verbose=True):
    x = cp.Variable(A.shape[1])
    n = A.shape[0]
    w = np.ones(n)
    r = (A @ x - b)
    f_r = f(r)

    if f_r.is_scalar():
      objective = cp.Minimize(f_r)
    else:
      objective = cp.Minimize(w @ f_r)
    prob = cp.Problem(objective)
      
    return prob, x 


def S_optimalizuj(A, b, f=cp.sum_squares,
                verbose=False):
    prob, x = S_vytvor_model(A, b, f)
    prob.solve(solver=cp.CLARABEL, verbose=verbose,
                  reduced_tol_gap_abs=10000.0,
                  reduced_tol_gap_rel=10.0,
                  reduced_tol_feas=20.0)
    return x.value


def S_logisticka_regresia(A, b, lam = 0.0, verbose = True):
    y = (b>0).astype(float)   # y={0,1}
    x = cp.Variable(A.shape[1])
    
    z = A @ x
    loss = cp.sum(cp.logistic(z) - cp.multiply(y, z))  
    objective = cp.Minimize(loss + lam * cp.norm(x, 1))
    
    problem = cp.Problem(objective)
    problem.solve(solver=cp.ECOS, verbose=verbose)    
    #problem.solve(solver=cp.CLARABEL, verbose=verbose,
                  #reduced_tol_gap_abs=10000.0,
                  #reduced_tol_gap_rel=10.0,
                  #reduced_tol_feas=20.0), reduced_tol_gap_abs=1e-2)
    return x.value

#---------------------------------------------------
# Optimalne vahy pre inverzny hyperbolicky sinus (S)
#---------------------------------------------------
def S_optimalne_v(zoznam):
    periody = zoznam["perioda"].unique()
    values_S = []
    
    for per in tqdm(periody):
        perioda = per
        A, b = vytvor_datasety(zoznam, perioda)
        A = np.asinh(A)
        b = np.asinh(b)
        
        norma_2 = S_optimalizuj(A, b, f = lambda r: cp.norm(r,2))
        norma_1 = S_optimalizuj(A, b, f = lambda r: cp.norm(r,1))  
        norma_max = S_optimalizuj(A, b, f = lambda r: cp.norm(r,'inf'))
        

        penal = S_optimalizuj(A, b, f = lambda r: cp.sum(cp.abs(r) ** 1.5))  
        huber = S_optimalizuj(A, b, f = lambda r: cp.sum(cp.huber(r,50))) 
        
        # logisticka regresia
        log_reg = S_logisticka_regresia(A, b, lam = 0.0, verbose=True)
        value = [perioda, norma_2, norma_1, norma_max, penal, huber, log_reg]
        values_S.append(value)
        
    stplec_S = ["perioda","norma_2","norma_1","norma_max", "penal","huber","log_reg"]
    optimalne_vahy_S = pd.DataFrame(data = values_S, columns=stplec_S)
    print("Optimalne vahy kazdej metody: ")
    print(optimalne_vahy_S)
    optimalne_vahy_S.to_excel("...\\data\\optimalne_vahy_S.xlsx",sheet_name= "Sheet1",index=True)   

    return optimalne_vahy_S

def S_sprav_excel_tabulku(zoznam_2, optimalne_vahy):
    probl = []
    row_w = []
    start = datetime.datetime(2025, 9, 1, 0, 17)
    for i in tqdm(range(len(zoznam_2))):
      current = start + datetime.timedelta(minutes=i)
      cas = current.strftime("%d.%m %H:%M")

      odchylka = (zoznam_2.iloc[i,1]) * 4 # *4 aby sme mali [MW]
      odchylka_r = zoznam_2.iloc[i,1]
      perioda = zoznam_2.iloc[i,0]     
 
      
      #ziskava hodnoty
      A = zoznam_2.iloc[i,2:17].tolist()
      A_row   = np.asinh(np.array(A)) 
      row_w = optimalne_vahy.loc[optimalne_vahy["perioda"] == perioda].iloc[0]

      predikcie = [
            np.sinh(np.dot(A_row, row_w["norma_2"])),
            np.sinh(np.dot(A_row, row_w["norma_1"])),
            np.sinh(np.dot(A_row, row_w["norma_max"])),
            np.sinh(np.dot(A_row, row_w["penal"])),
            np.sinh(np.dot(A_row, row_w["huber"])),
            float(np.dot(A_row, row_w["log_reg"]))  # bez sinh
        ]
      value = [cas,perioda] + A_row.tolist() + predikcie + [odchylka, odchylka_r]
      probl.append(value)
    stlp = ["cas","model"] + [f"data_{k+1}" for k in range(15)] + \
        ["dvojkova norma","jednotkova norma","maximova norma","penalizacna funkcia",
         "huberova funkcia"," logisticka regresia", "realne odchylka [MW]","realne odchylka [MWh]"] 
    excel = pd.DataFrame(data = probl, columns=stlp)
    print("Predikcie odchylky nasimi metodami")
    print(excel)
    excel.to_excel("...\\data\\test_S.xlsx",sheet_name= "Sheet1",index=True)   
    return excel
   
#--------------------  
# Metriky modelov (S)
#--------------------
def S_vyhodnotenie(zoznam,optimalne_vahy):
    modely = ["norma_2", "norma_1", "norma_max", "penal", "huber","log_reg"]
    vysledky = {} 
    periody = zoznam["perioda"].unique()
    

    
    for perioda in tqdm(periody):
        per = zoznam[zoznam["perioda"] == perioda]
        row = optimalne_vahy.loc[optimalne_vahy["perioda"] == perioda].iloc[0]
        b = (per["odchylka"].values)[:-3] * 4  #posledne 3 su Nan *4 aby sme mali [MW]
        A = per.iloc[:-3, 2:17].values  
        A_new = np.asinh(A)
        
        perioda_metrics = {}
        
        for model in modely:
            vaha_modelu = row[model]
            
            if model == "log_reg":
                scores = A_new @ vaha_modelu
                

                y_pred = (scores > 0).astype(int)
                y_true = (b > 0).astype(int)
                

                accuracy = np.mean(y_pred == y_true)
                

                tp = np.sum((y_pred == 1) & (y_true == 1))  # True Positive
                tn = np.sum((y_pred == 0) & (y_true == 0))  # True Negative
                fp = np.sum((y_pred == 1) & (y_true == 0))  # False Positive
                fn = np.sum((y_pred == 0) & (y_true == 1))  # False Negative
                
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                

                perioda_metrics[f"{model}_accuracy"] = accuracy
                perioda_metrics[f"{model}_precision"] = precision
                perioda_metrics[f"{model}_recall"] = recall
                perioda_metrics[f"{model}_f1"] = f1
                
                # skutocne zaporne, predikcia kladna
                fn_znam = np.sum((b < 0) & (y_pred > 0))
                # skutocne kladne, predikcia zaporna  
                fp_znam = np.sum((b > 0) & (y_pred < 0))
                celkom  = len(b)
                
                print(f"Perioda {perioda:3d} | {model:10s} | "
                      f"zap-klad: {fn_znam:3d} ({100*fn_znam/celkom:.1f}%) | "
                      f"klad-zap: {fp_znam:3d} ({100*fp_znam/celkom:.1f}%)")
            else:
                pred = np.sinh(A_new @ vaha_modelu)  
                r    = pred - b 
    
                mse = np.mean(r**2)
                wape = np.sum(np.abs(r)) / np.sum(np.abs(b) + 1e-12)
                ss_res = np.sum(r**2)
                ss_tot = np.sum((b - np.mean(b))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                
    
                perioda_metrics[f"{model}_Mse"] = mse
                perioda_metrics[f"{model}_Wape"] = wape
                perioda_metrics[f"{model}_R2"] = r2
            

        vysledky[perioda] = perioda_metrics
        
    df_vysledky = pd.DataFrame.from_dict(vysledky , orient='index') 
    with pd.ExcelWriter("...\\data\\test_S.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:
         df_vysledky.to_excel(writer, sheet_name="Sheet2", index=True)
    print("metriky nasich metod: ")
    print(df_vysledky)
    
    return df_vysledky

def program():
    # Trenovancie data zlozene z 11 mesiacov 
    zoznam = nacitanie_dat("...\\data\\Trenovacie_data.xlsx")
    # testovacie data zlozene z jedneho mesiaca (September)
    zoznam_2 = nacitanie_dat("...\\data\\2025-09.xlsx")
    
    

    # Poistenie (P)
    # opt_vahy_x = pd.read_excel("...\\data\\optimalne_vahy_P.xlsx")
    # for col in ["norma_2", "norma_1", "norma_max", "penal", "huber", "log_reg"]:
    #     opt_vahy_x[col] = opt_vahy_x[col].apply(
    #         lambda s: np.fromstring(s.strip("[]"), sep=" ") if isinstance(s, str) else s)
    
    # Povodny dataset    
    opt_vahy_x = optimalne_v(zoznam)
    sprav_excel_tabulku(zoznam_2, opt_vahy_x)
    vyhodnotenie(zoznam_2, opt_vahy_x)
    
    # Poistenie (A)
    # opt_vahy_x_A = pd.read_excel("...\\data\\optimalne_vahy_A.xlsx")
    # for col in ["norma_2", "norma_1", "norma_max", "penal", "huber", "log_reg"]:
    #     opt_vahy_x_A[col] = opt_vahy_x_A[col].apply(
    #         lambda s: np.fromstring(s.strip("[]"), sep=" ") if isinstance(s, str) else s)
    
    # Rozsireny dataset o absolutne hodnoty 
    opt_vahy_x_A = A_optimalne_v(zoznam)
    A_sprav_excel_tabulku(zoznam_2, opt_vahy_x_A)
    A_vyhodnotenie(zoznam_2, opt_vahy_x_A)
    
    
    # Poistenie (S)
    # opt_vahy_x_S = pd.read_excel("...\\data\\optimalne_vahy_S.xlsx")
    # for col in ["norma_2", "norma_1", "norma_max", "penal", "huber", "log_reg"]:
    #     opt_vahy_x_S[col] = opt_vahy_x_S[col].apply(
    #         lambda s: np.fromstring(s.strip("[]"), sep=" ") if isinstance(s, str) else s)
        
    # Transformovany dataset pomocou inverzneho hyperbolickeho sinusu
    opt_vahy_x_S = S_optimalne_v(zoznam)
    S_sprav_excel_tabulku(zoznam_2, opt_vahy_x_S)
    S_vyhodnotenie(zoznam_2, opt_vahy_x_S)
    
    

program()

#======
# Grafy
#======


#-------------------------
# Graf pre AUC-ROC (A,P,S)
#-------------------------

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

zoznam_2    = nacitanie_dat("...\\data\\2025-09.xlsx")
# Povodny dataset (P)
#opt_vahy_x  = pd.read_excel("...\\data\\optimalne_vahy_P.xlsx")

# Rozsireny dataset (A)
# opt_vahy_x  = pd.read_excel("...\\data\\optimalne_vahy_A.xlsx")

#  Transformovany dataset (S)
# opt_vahy_x  = pd.read_excel("...\\data\\optimalne_vahy_S.xlsx")

for col in ["norma_2", "norma_1", "norma_max", "penal", "huber", "log_reg"]:
    opt_vahy_x[col] = opt_vahy_x[col].apply(
        lambda s: np.fromstring(s.strip("[]"), sep=" ") if isinstance(s, str) else s)
    
    

all_scores = []
all_labels = []

periody = sorted(zoznam_2["perioda"].unique())
n = len(periody)


fig, ax = plt.subplots(figsize=(9, 7))
for idx, perioda in enumerate(periody):
    
    per = zoznam_2[zoznam_2["perioda"] == perioda]
    row = opt_vahy_x.loc[opt_vahy_x["perioda"] == perioda]
    b = per["odchylka"].values[:-3] * 4  # *4 aby sme mali [MW]
    
    # Povodny dataset (P)
    #A = per.iloc[:-3, 2:17].values
    
    # Rozsireny dataset (A)
    #A = np.hstack([A, np.abs(A)])
    
    # Transformovany dataset (S)
    #A = np.asinh(A)
    
    weight = row.iloc[0]["log_reg"]
    scores  = A @ weight
    labels  = (b > 0).astype(int)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc     = auc(fpr, tpr)

    is_first = (idx == 0)
    is_last  = (idx == n - 1)

    if is_first or is_last:
        color = "red"
        lw = 2.5
        zorder = 5
        tag = "prvá" if is_first else "posledná"
        label = f"Submodel {perioda} ({tag})  AUC={roc_auc:.3f}"
    else:
        color = plt.cm.Blues(0.4 + 0.5 * idx / n)
        lw = 1.2
        zorder = 2
        label = f"Submodel {perioda}  AUC={roc_auc:.3f}"

    ax.plot(fpr, tpr, color=color, lw=lw, zorder=zorder, label=label)
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Náhodný klasifikátor")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate ", fontsize=12)
ax.set_ylabel("True Positive Rate",      fontsize=12)
ax.legend(loc="lower right", fontsize=8.45, ncol=1)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(f"...\\data\\log_reg_S.png", dpi=150)
plt.show()


#------------------------------------------------------------------
# Graf pre porovnanie Predikovanej vs. Realnej odchylky dataset (P)
#------------------------------------------------------------------

zoznam_2    = pd.read_excel("\\test_P.xlsx")
opt_vahy_x  = pd.read_excel("\\optimalne_vahy_P.xlsx")

for col in ["norma_2", "norma_1", "norma_max", "penal", "huber"]:
    opt_vahy_x[col] = opt_vahy_x[col].apply(
        lambda s: np.fromstring(s.strip("[]"), sep=" ") if isinstance(s, str) else s)
    
  
def vykresli_graf(zoznam_2, metoda, model):
    predikcia  = zoznam_2[metoda].values[1183:1363]
    skutocnost = zoznam_2["realne odchylka [MW]"].values[1183:1363]
    #priebezna =  zoznam_2["data_15"].values[1183:1363]
    n = len(predikcia)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.step(range(n), skutocnost, where="post", color="darkorange",  lw=2, label="systémová odchýlka")
    #ax.step(range(n), priebezna, where="post", color="royalblue", lw = 1.6, label="priebežná odchýlka")
    ax.step(range(n), predikcia,  where="post", color="royalblue",   lw=1.2, label="predikcia")
    ax.axhline(0, color="red", lw=1)

    ax.set_title(model)
    ax.set_xlabel("Číslo minúty")
    ax.set_ylabel("Hodnota [MW]")
    ax.set_xlim(0, n)
    ax.set_xticks(np.arange(0, n + 1, 15))
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"...\\data\\graf_{model}.png", dpi=150)
    plt.show()   


mod =   ["dvojkova norma",
         "jednotkova norma",
         "maximova norma",
         "penalizacna funkcia",
         "huberova funkcia",
         ]

modely = ["Euklidovská norma",
         "Manhattanská norma",
         "Čebyševova norma",
         "Norma $\\ell_{1{,}5}$",
         "Huberova funkcia",
         "logisticka regresia"
         ] 

for m,j in zip(mod, modely):
    vykresli_graf(zoznam_2,m,j)     
    
    
    


r = len(zoznam_2.iloc[:,23])
values = []
for i in range(r):
    score = zoznam_2.iloc[i,23]
    p = 1/(1 + np.exp(-score))
    value = p
    values.append(value) 
zoznam = pd.DataFrame(data = values)
zoznam.to_excel("...\\data\\Upravene_data_09.xlsx",index=False)   
print(zoznam)

#--------------------------------------------------
# Graf pre logisticku regresiu a priebeznu odchylku
#--------------------------------------------------
zoznam_2 = pd.read_excel("...\\data\\test_A.xlsx")  
    
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


score = zoznam_2["logisticka regresia"].values[1183:1363]
pravdepodobnost = 1/(1 + np.exp(-score))
skutocnost = zoznam_2["realne odchylka [MW]"].values[1183:1363]
n = 180
fig, ax1 = plt.subplots(figsize=(10, 4))

# Lava os: skutocna odchylka
ax1.step(range(n), skutocnost, where="post", color="darkorange", lw=1.5, label="Skutočná odchýlka")
ax1.axhline(0, color="red", lw=1.2, linestyle="-", label="Nulová odchýlka / hranica znamienka")
ax1.set_xlabel("Číslo minúty")
ax1.set_ylabel("Systémová odchýlka [MW]", color="darkorange")
ax1.tick_params(axis="y", labelcolor="darkorange")

# Prava os: pravdepodobnost
ax2 = ax1.twinx()
ax2.step(range(n), pravdepodobnost, where="post", color="royalblue", lw=1.0, label="P(odchýlka > 0)")
ax2.set_ylabel("Pravdepodobnosť kladnej odchýlky", color="royalblue")
ax2.tick_params(axis="y", labelcolor="royalblue")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax2.set_ylim(0, 1) 
max_val = max(abs(skutocnost))
ax1.set_ylim(-max_val, max_val)

for k in range(0, n, 15):
    ax1.axvline(k, color="gray", lw=0.6, linestyle="--", alpha=0.5)
ax1.set_xticks(np.arange(0, n + 1, 15))
ax1.set_xlim(0, n) 
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
plt.title("Logistická regresia — skutočná odchýlka vs. pravdepodobnosť znamienka")
plt.tight_layout()
plt.savefig("...\\data\\graf_logisticka_regresie.png", dpi=150)  
plt.show()
 
#---------------------------------
# Graf inverzny hyperbolicky sinus
#---------------------------------
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-250, 250, 10000)
y = np.arcsinh(x)

plt.figure(figsize=(10, 4))
plt.plot(x, y, color="royalblue", linewidth=2)
plt.title("Inverzný hyperbolický sínus")
plt.xlabel("x")
plt.ylabel("arcsinh(x)")
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(np.arange(-500, 500, 100))
plt.xlim(-500, 500)
plt.tight_layout()
plt.savefig("...\\data\\graf_asinh.png", dpi=150)
plt.show()

#----------------------------------
# Graf Huberova penalizacna funkcia
#----------------------------------

import numpy as np
import matplotlib.pyplot as plt

def huber(x, M):
    abs_x = np.abs(x)
    return np.where(abs_x <= M,
                    x**2,
                    2 * M * abs_x - M**2)

x = np.linspace(-250, 250, 10000)
y = huber(x, 50)

plt.figure(figsize=(10, 4))
plt.plot(x, y, color="royalblue", linewidth=2)
plt.title("Huberova penalizačná funkcia (M = 50)")
plt.xlabel("x")
plt.ylabel("huber(x)")
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(np.arange(-250, 251, 50))
plt.xlim(-250, 250)
plt.tight_layout()
plt.savefig("...\\data\\graf_Huber.png", dpi=150)
plt.show()

#----------------------
# Graf spotreby fabriky 
#----------------------
import numpy as np
import matplotlib.pyplot as plt


fabrika = pd.read_excel("...\\data\\Upravene_data_09.xlsx")

spotreba = fabrika.iloc[:,0] 
x = np.arange(len(spotreba)) 

fig, ax = plt.subplots(figsize=(10, 4))
ax.step(x, spotreba, where="post", color="green", lw=2, label="Spotreba")
ax.set_title("Spotreba fabriky v každej minúte počas celého testovacieho obdobia")
ax.set_xlabel("Číslo minúty")
ax.set_ylabel("Hodnota [MW]")
ax.set_xlim(0, len(spotreba) - 1)
ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("...\\data\\graf_fabrika.png", dpi=150)
plt.show()

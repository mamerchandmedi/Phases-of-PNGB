from gegenbauer_lib import *


def myfun(somearg):
    dict_out={} ###Here is the output
    np.random.seed()
    numG=3 ###Number of Gegenbauer plynomials
    nvec=sorted(np.random.randint(1,100,numG))
    print("Order of the polynomials are:",nvec)
    if (len(nvec)!=set(nvec))==False: ###Checks if there are not repeated element in nvec
        print("Repeated polynomial")
        return
    avec=sorted(np.random.uniform(-10,10,numG),key=abs)
    print("Coefficients are:",avec)
    for i in range(1,len(nvec)+1):
        dict_out.update({"n"+str(i):nvec[i-1],"a"+str(i):avec[i-1]})
    ##Set up the model
    myargs={"Ngb":4,"na_coeffs":list(np.array((nvec,avec)).T)}
    m=model1(myargs)
    m.include_CW=False
    if m.f==np.inf:
        print("Model does not work")
        return dict_out
    isEWSB=m.isEWSB()
    if ((isEWSB[1][0]-v)<5):
        dict_out.update({"EWSB":True})
    else:
        dict_out.update({"EWSB":False})
        return dict_out
    m.Higgs_trilinear()
    dict_out.update({"chh":m.chhh})
    m.findTrestoration()
    if m.Trestored==None:
        dict_out.update({"Tres":None})
        return dict_out
    dict_out.update({"Tres":m.Trestored})
    m.minTracker()
    m.findTcritical()
    if len(m.Tc)==0:
        return dict_out
    m.findAllTransitions()
    for i in range(len(m.TnTrans)):
        dict_out.update(m.TnTrans[i])


    return dict_out



###Do parallelization

from multiprocessing import Pool
import time
start = time.time()



##The Multiprocessing package provides a Pool class,
##which allows the parallel execution of a function on the multiple input values.
##Pool divides the multiple inputs among the multiple processes which can be run parallelly.
num_points=20
f= myfun
if __name__ == '__main__':
    with Pool() as p:
        df_pool=p.map(f, range(num_points))



print(df_pool)
pd.DataFrame(df_pool).to_csv("./scan_1.csv")



end = time.time()
print("The time of execution of above program is :", end-start)

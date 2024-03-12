import sys
import numpy as np
import scipy.sparse as sp
import numpy as np
import cvxpy as cp
class Caculate_parameters_conv():

    def __init__(self, number_neighbors, seq_input_x, max_allow_spatial_conv = 2, max_allow_dilation = 2, weight = 'std', total_number = 8):
        super(Caculate_parameters_conv, self).__init__()
        self.number_neighbors = number_neighbors
        self.seq_input_x = seq_input_x
        self.max_allow_spatial_conv = max_allow_spatial_conv
        self.max_allow_dilation = max_allow_dilation
        self.weight =  weight
        self.total_number = total_number

    def main(self):
        w = []
        b = []
        v = []
        bags = self.constructed_bags()
        print(" bags ", bags)
        for bag in bags:
            w.append(bag[4])
            b.append(bag[5])
            if self.weight == 'std':
                v.append(- np.array([bag[4],bag[5]]).std())
            elif  self.weight == 'mean':
                v.append(- np.array([bag[4], bag[5]]).mean())

        x_list = self.interger_programming_conv(w,b,v,self.number_neighbors - 1, self.seq_input_x - 1 , total_number = self.total_number )
        # print("total_number,", x_list)
        final_convs = []
        for index,x in enumerate(x_list):
            x = int(x)
            [final_convs.append(bags[index]) for i in range(x)]
        # np.random.shuffle(final_convs)
        return final_convs



    def constructed_bags(self):
        bags = []
        for ker1 in range(1,self.max_allow_spatial_conv+1):
            for ker2 in range(1,self.max_allow_spatial_conv+1):
                for dila1 in range(1, self.max_allow_dilation + 1):
                    for dila2 in range(1, self.max_allow_dilation + 1):
                        if ker1 < dila1 or ker2 < dila2:
                            continue
                        weig1 = dila1 * (ker1 - 1)
                        weig2 = dila2 * (ker2 - 1)
                        bag = (ker1, ker2, dila1, dila2, weig1, weig2)
                        bags.append(bag)

        def compare_func(x):
            return  -x[-1]*x[-1] - x[-2]*x[-2]
        bags = sorted(bags, key = compare_func, reverse=True)
        return bags

    def interger_programming_conv(self, w, b, v, neighbors, seq_in, total_number=8):
        n = len(w)

        c = np.array(v)

        a = np.array([w , b]).reshape(2,-1)

        b = np.array([neighbors,seq_in])

        x = cp.Variable(n, integer=True)

        objective = cp.Maximize(cp.sum(c * x))

        constriants = [0 <= x, a * x == b, cp.sum(x)==total_number]

        prob = cp.Problem(objective, constriants)

        resluts = prob.solve(solver=cp.CPLEX)

        return x.value


if __name__ == '__main__':
    for i in range(1):
        main = Caculate_parameters_conv(1,12,total_number = 8)
        result = main.main()
        result = np.array(result)
        print("---", result)
        # print("---",result[:,:2],result[:,2:4])
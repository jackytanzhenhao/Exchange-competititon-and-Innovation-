import unittest
import jax
from itertools import product


class Test(self):
    def setUp(self):
        self.key = jax.random.PRNGKey(0)
        self.J = jax.random.truncated_normal(self.key, -1, 1, shape = (100,))

    
    def TestNvalue(self):
        N = 2
        p_pri = 0.3
        p_pub = 0.2
        l_inv = 0.5
        with self.assertRaise(ValueError):
            s_star(self.J, p_pri,p_pub,l_inv,N)



    def Test_general(self):
        #testing the general relationship that s_n is smaller than s_continuous 
        J = jax.random.truncated_nomral(self.key, -1, 1)
        N = 3
        # create probability simplex
        grid = product(range(0,1,0.1),3) 
        probsimplex = [point for point in grid if sum(point) == 1]

        for i in probsimplex:
            with self.subTest(prob = i ):
                self.assertTrue(s_star(J, prob[0],p_pub,l_inv,N) > s_lw(J, p_pri, p_pub,l_inv, N))


    
    

        


#class Test_s_lw(self):

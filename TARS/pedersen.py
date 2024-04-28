import random
from math import ceil, gcd
from decimal import Decimal
from Crypto.Util import number

global g, h, p, q

class Pedersen():

    def __init__(self) -> None:
        self.p = -1
        self.q = number.getPrime(1024)
        self.g = self.getGenerators(self.q)
        self.h = self.getHval(self.g, self.q)
        self.k=20


    def vss(self, secret, shareP):
        # coefficient of s and t
        coeffsS = self.coeff(self.k, secret)
        coeffsT = self.coeff(self.k, random.randrange(0, self.q))

        # generate shares
        shares = self.generateShares(coeffsS, coeffsT, shareP, self.k)

        # genrate public commitments
        sharedCommits = self.sharedCommit(coeffsS, coeffsT, self.k)
        return shares, sharedCommits

    def recon(self,shareP,shares,sharedCommits):

        # verify shares
        for i in range(shareP):
            #print("Verification for aggregator No. " + str(i + 1) + ": ",
            self.verifySecret(shares[i][0], shares[i][1], sharedCommits, i + 1)
        #print("\n")

        # reconstruct secret using k shares
        # kshares = random.sample(shares, k)
        gensecret = self.reconstructSecret(shares, self.k)
        return gensecret


    # check if no. is prime
    def isPrime(self,n):
        if (n <= 1):
            return False
        if (n <= 3):
            return True
        if (n % 2 == 0 or n % 3 == 0):
            return False
        i = 5
        while(i * i <= n):
            if (n % i == 0 or n % (i + 2) == 0):
                return False
            i = i + 6
        return True

    # prime nos generator
    def getPrimes(self,n):
        k = number.getPrime(n)  # n-bit random prime no
        s = k-1
        while((k-1) % s != 0 or not self.isPrime(s)):
            s -= 1
        return k, s

    # power in O(|y|)
    def power(self,x, y, z):
        res = 1
        x = x % z
        if (x == 0):
            return 0
        while (y > 0):
            if ((y & 1) == 1):
                res = (res * x) % z
            y = y >> 1
            x = (x * x) % z
        return res

    # get 'g' a generator
    def getGenerators(self,n):
        for i in range(n-1, 1, -1):
            if (gcd(i, n) == 1):
                return i

    # cal h
    def getHval(self,g, n):
        res = g
        while(res == g):
            res = random.randrange(0, self.q)
        return res

    # calc random coeficient
    def coeff(self,k, secret):
        coeff = [secret]
        for i in range(k-1):
            coeff.append(random.randrange(0, self.q))
        return coeff

    # calc y-coordinate
    def calcY(self,x, coeffs, k):
        y = 0
        for i in range(k):
            y += x**i * coeffs[i]
        return y

    # split secret between shareholders
    def generateShares(self,coeffsS, coeffsT, shareno, k):
        shares = []
        for x in range(1, shareno+1):
            shares.append([self.calcY(x, coeffsS, k), self.calcY(x, coeffsT, k)])
        return shares

    #  commitment scheme
    def commitment(self,s, t):
        return self.power(self.g, s, self.q) * self.power(self.h, t, self.q)

    # shared commitments
    def sharedCommit(self,coeffsS, coeffsT, k):
        sharedCommit = []
        for i in range(k):
            sharedCommit.append(self.commitment(coeffsS[i], coeffsT[i]) % self.q)
        return sharedCommit

    # verification
    def verifySecret(self,s, t, sharedCommit, i):
        currCommit = self.commitment(s, t) % self.q
        combCommit = sharedCommit[0]  % self.q
        #combCommit = (sharedCommit[0] * (power(sharedCommit[1], i, q))) % q
        for l in range(1,self.k):
            combCommit*= self.power(sharedCommit[l], i ** l, self.q) % self.q
        return currCommit == (combCommit % self.q)

    # reconstruct secret
    def reconstructSecret(self,shares, k):
        a0_reconstructed = 0
        for i in range(1, k + 1):
            #print("Share of P_" + str(i) + " is " + str(shares[i - 1][0]))
            a0_reconstructed += self.delta(i) * shares[i - 1][0]
        return a0_reconstructed

    def delta(self,i):
        d = 1
        #print("\ni= ")
        #print(i)
        for j in range(1,self.k+1):
            if j != i:
                d *= -j / (i - j)
                #d=j
        #print("delta = " + str(d))
        return int(d)
from sage.all import *
from sage.calculus.predefined import x
from cryptography.hazmat.primitives import hashes
from collections import namedtuple

# Parameters Credit: https://github.com/blynn/pbc - Ben Lynn, 2006
# Parameters of Type D curve : 
#    q : base field size
# a, b : curve parameters
#    n : group order on curve
#    r : torsion group size
#    k : embedding degree

q = 15028799613985034465755506450771565229282832217860390155996483840017
a = 1871224163624666631860092489128939059944978347142292177323825642096
b = 9795501723343380547144152006776653149306466138012730640114125605701

n = 15028799613985034465755506450771561352583254744125520639296541195021
r = 15028799613985034465755506450771561352583254744125520639296541195021
k = 6

F = GF(q ** 6, modulus=x**6+x+1, name='a')
E = EllipticCurve(F, [a, b])
Frob = [F.frobenius_endomorphism(i) for i in range(k)]

R = namedtuple("Member", "public_key public_id")

Ring = []

def pairing(e1, e2):
  return e1.weil_pairing(e2, r)

class GenPP:

  # 生成用于配对和加密的公共参数
  # g1 : generator of G1 group
  # g2 : generator of G2 group
  # g3 : generator of Gt group

  def Trace(self, P):
    Q = P
    
    for i in range(1,k):
      X = Frob[i](P[0])
      Y = Frob[i](P[1])
      Q = Q + E(X, Y)
    
    return Q

  def __init__(self) -> None:

    ord = E.order() / r ** 2

    g = E.random_point()

    while (g * ord).is_zero():
      g = E.random_point()

    g = g * ord

    self.g1 = self.Trace(g)
    self.g2 = k * g - self.g1

    self.g3 = pairing(self.g1, self.g2)

    self.ModRing = IntegerModRing(r)

  # return a point in G1 by computing g1 ^ index

  def G1(self, index = 1):
    return self.g1 * index


  def Gt(self, index = 1):
    return pairing(self.g1 * index, self.g2)

  # 返回一个确定性的散列到Z/rZ
  # 元素可以是  
  # - 一个字节的字符串, 或
  # - 一个有限域元素, 或
  # - 曲线上的R扭转点

  def Zr_hash(self, element):

    try:

      element.decode()
      message = element

    except AttributeError:

      try:

        element = element.xy()[0]

      except:

        pass
    
      coef_str = '.'.join(map(str, element.polynomial().coefficients()))
      message = coef_str.encode()

    digest = hashes.Hash(hashes.SHA224())
    digest.update(message)

    return self.ModRing(int.from_bytes(digest.finalize(), 'big'))

  # return a random element from Z/rZ

  def RandInt(self):
    return self.ModRing.random_element()

pp = GenPP()

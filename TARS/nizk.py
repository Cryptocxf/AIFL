from sage.all import *
from keygen import *

# NIZK{r: a^r = b}

def schnorr_proof(r, a):

  u = pp.RandInt()
  t = a ** u
  c = pp.Zr_hash(t)
  z = u + r * c

  return (t, z)

def schnorr_verify(a, b, resp):

  t, z = resp
  c = pp.Zr_hash(t)

  return a ** z == b ** c * t

# 通过检查来验证签名的正确性:
# 1. 对每个环内成员的签名密钥进行正确加密;
# 2. 根据验证密钥和追踪者的密钥对公共身份进行正确加密

# !!! 任何第三方（不一定是环成员）都应该能够验证签名。
# 不透露哪个成员产生了这种签名,
# 因此，在环中确保了匿名性。
# NIZK_Verify(pp,(pk_sign,(pk_PKE_1,...,pk_PKE_|R|),c_1),rou)=0


# 用Schnorr的协议生成一个模拟,
def schnorr_simulate(g, u, c):

  z = pp.RandInt()
  t = g ** z / u ** c

  return (t, z)

def schnorr_sim_verify(a, b, t, c, z):
  #print('az=',a ** z)
  #print('bct=',b ** c * t)

  return a ** z == b ** c * t

# 用Schnorr的协议生成一个模拟,
def okamoto_simulate(g, u, c):

  r = pp.RandInt()
  t = g ** r / u ** c

  return (t, r)

def okamoto_sim_verify(g, u, t, c, z):
  #print('gz',g ** z)
  #print('uct',u ** c * t)
  return g ** z == u ** c * t



# 1. 证明签字者是环的成员
# 2. 证明签名者自己的公共ID的正确加密
# {(sk, r2, r3, i) : g3 ^ sk = PID_i and PKtr ^ r2 * PKsign ^ r3 * PID_i = c3}


def signature_of_knowledge_proof(index, secret_key, r2, PKtr, PID_encryption, message):

  commit_schnorr , commit_okamoto = ([0] * len(Ring) , [0] * len(Ring))
  challenge = []
  response_schnorr , response_okamoto = ([0] * len(Ring) , [0] * len(Ring))

  c_sum = 0
  #print(message)
  c = pp.Zr_hash(message)
  #print(c)

  for i in range(len(Ring)):

    challenge.append(Integer(pp.RandInt()))
    commit_schnorr[i], response_schnorr[i] = (schnorr_simulate(pp.g3, Ring[i].public_id, challenge[i]))
    commit_okamoto[i], response_okamoto[i] = (okamoto_simulate(PKtr, PID_encryption[1]/ Ring[i].public_id, challenge[i]))

    if i != index:
      c_sum = c_sum ^ challenge[i]
      c *= pp.Zr_hash(commit_schnorr[i])*pp.Zr_hash(commit_okamoto[i])

  u = pp.RandInt()

  commit_schnorr[index] = pp.g3 ** u
  commit_okamoto[index] = PKtr ** u

  c *= pp.Zr_hash(commit_schnorr[index]) * pp.Zr_hash(commit_okamoto[index])

  challenge[index] = Integer(c) ^ c_sum

  response_schnorr[index] = secret_key * challenge[index] + u
  response_okamoto[index] = r2 * challenge[index] + u

  return [(commit_schnorr,commit_okamoto), challenge[:-1], (response_schnorr,response_okamoto)]

#SoK.Verify(pp,(pk_T,pk_sign,R,c_3),m,sigma_SoK)=0
def signature_of_knowledge_verify(signature, PKtr, message):

  PID_encryption, PID_signature = signature
  (commit_schnorr,commit_okamoto), challenge, (response_schnorr,response_okamoto) = PID_signature

  c = pp.Zr_hash(message)

  for com in commit_schnorr: 
    c *= pp.Zr_hash(com) 

  for com in commit_okamoto:
    c *= pp.Zr_hash(com)

  c = Integer(c)

  for ch in challenge:
    c = c ^ ch
  
  challenge.append(c)


  return all([
    schnorr_sim_verify(pp.g3, Ring[i].public_id, commit_schnorr[i], challenge[i], response_schnorr[i])
    for i in range(len(Ring))
  ]) and all([
    okamoto_sim_verify(PKtr, PID_encryption[1] / Ring[i].public_id, commit_okamoto[i], challenge[i], response_okamoto[i])
    for i in range(len(Ring))
  ])

def trace_verify(PKtr, message, signature, PID, proof_of_trace):

  if not signature_of_knowledge_verify(signature,PKtr, message):

    print("Signature forgery detected!")
    return 0

  PID_encryption, PID_proof = signature
  #PID_encryption, PID_proof = PID_encryption_proof
  #tr, SKsign = trace

  if not all([
    schnorr_verify(pp.g3, PKtr, proof_of_trace[0]),
    schnorr_verify(PID_encryption[0], PID_encryption[1]/ PID, proof_of_trace[1])]):
    
    print("Proof of trace didn't pass!")
    return 0

  return 1

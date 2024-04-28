from sage.all import *
from nizk import *
from keygen import *

class Tracer():

  # 为Tracer生成一个随机密钥对
  # (PKt, SKt) in (Gt, Z/rZ)


  def __init__(self) -> None:
    self.__secret_key = pp.RandInt()
    self.public_key = pp.Gt(self.__secret_key)

    #print('self.__secret_key:',self.__secret_key)
    #print('self.public_key:',self.public_key)

  # 追踪器首先验证签名,
  # 然后解密签名者的公共身份信息
  # 并生成一个正确解密的证明
  def trace(self, message, signature):

    try:
      if not signature_of_knowledge_verify(signature,self.public_key, message):
        return 0
    except:
      raise("Signature verification issue!")

    #key_encryption_proof, signature_SoK = signature
    PID_encryption, PID_signature = signature

    PID = PID_encryption[1] / (PID_encryption[0] ** self.__secret_key)
    #print('PID=',PID)

    proof_of_trace = [
      schnorr_proof(self.__secret_key, pp.g3),
      schnorr_proof(self.__secret_key, PID_encryption[0])
    ]
    
    return (PID, proof_of_trace,self.__secret_key)

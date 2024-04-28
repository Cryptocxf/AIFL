from sage.all import *
from nizk import *
from keygen import *

class User:

  # 为每个环用户生成一个随机密钥对
  def __init__(self) -> None:
    self.__secret_key = pp.RandInt()
    self.public_key = pp.G1(self.__secret_key)
    self.public_id = pp.Gt(self.__secret_key)
    self.index = len(Ring)
    Ring.append(R(self.public_key, self.public_id))

  # 在追踪者的公钥下加密自己的公有身份
  # 和自己的签名密钥，并生成一个正确的加密的签名

  def encrypt_PID_and_sign(self, PKtr, message):
    r2  = pp.RandInt()
    PID_encryption = [pp.g3 ** r2, PKtr ** r2 * self.public_id]
    PID_signature = signature_of_knowledge_proof(self.index, self.__secret_key, r2, PKtr, PID_encryption, message)
    return (PID_encryption, PID_signature) #c_3和sigma_Sok

  # 通过加密每个环成员的秘密签名密钥来签署一个信息（字节）
  # 并用公共签名密钥和追踪者的公钥对自己的公有身份进行加密
  # 用ZK证明来确保双方的正确加密.
  # Sign(pp,sk_U,pk_T,m,R)
  def sign(self, PKtr, message):

    signature = self.encrypt_PID_and_sign(PKtr, message)

    return  signature


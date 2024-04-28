from sage.all import *
from nizk import *
from keygen import Ring, pp
import user, tracer, keygen, pedersen
from time import time
import struct

if __name__ == "__main__":

  [user.User() for _ in range(0)]

  shareP = 30
  #print("Number of aggregators: ", shareP)
  #secret = 3451
  #print("The secret is: " + str(secret))

  user = user.User()
  tracer = tracer.Tracer()
  pedersen = pedersen.Pedersen()
  gradient = 0.5
  message = struct.pack('!f',gradient)
  clock = time()
  kengen = keygen.GenPP()

  print("kengen time:", time() - clock)

  clock = time()
  signature = user.sign(tracer.public_key, message)

  print("signing time:", time() - clock)
  clock = time()

  assert(signature_of_knowledge_verify(signature,tracer.public_key, message))
  print("signature verification time:", time() - clock)

  clock = time()

  PID, proof_of_trace,secret = tracer.trace(message, signature)
  print("trace time:", time() - clock)
  clock = time()

  assert(trace_verify(tracer.public_key, message, signature, PID, proof_of_trace))
  print("trace verification time:", time() - clock)

  clock = time()
  shares, sharedCommits = pedersen.vss(int(secret),shareP)
  #print(gensecret)

  print("share time:", time() - clock)

  clock = time()
  gensecret = pedersen.recon(shareP,shares, sharedCommits)
  #print(gensecret)

  print("recon time:", time() - clock)

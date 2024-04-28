# A Threshold-based Accountable Ring Signature (TARS):

A Sage Math implementation of the accountable ring signature protocol proposed in https://doi.org/10.1007/978-3-030-92548-2_10 and https://eprint.iacr.org/2022/1293.

## Dependencies:

https://github.com/pyca/cryptography/

## Test Run:

To test the correctness of the program, run

```
load('main.py')
```

in a Sage interactive shell.

## Zero-knowLedge Proofs

ZK proofs are introduced to ensure the integrity of protocol execution. They are summarized as follows:

1. Key Encryption Stage: The signer proves correct encryption of $PK_\text{sign}$.
2. Identity Encryption Stage: The signer proves his membership, and proves correct encryption of his own $PID_\text{u}$.
3. Reporting Stage: The reporter proves correct decryption of $SK_\text{sign}$. (However, this proof is not explicitly instantiated, as explained in the next section).
4. Tracing Stage: The tracer proves correct decryption of $PID_\text{u}$ using $SK_\text{sign}$ and $SK_\text{T}$.

## Requirement: 

- Sage version: 9.6 or above
- Python: 3.8


## Benchmark

Our experimental environment was set up in a virtual machine running on a 64-bit Ubuntu 20.04.5 LTS PC with an Intel CPU clocked at 2.50GHz$\times$8 and 15.6GiB RAM.
The benchmark is performed on a ring of 120 clients. The signature is generated for a single message of 16 bytes.

| function               | time (s) |
|------------------------|-------|
| Setup                  | 0.55  |
| Sign                   | 1.04  |
| Verify and VerTrace    | 0.922 |
| Trace                  | 1.324 |

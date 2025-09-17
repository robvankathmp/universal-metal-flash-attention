import Foundation
import MFABridge
@testable import MFAFFI

@inline(__always)
func mfa_attention_forward_nomask(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?,
  _ k: UnsafeMutableRawPointer?,
  _ v: UnsafeMutableRawPointer?,
  _ out: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ headDim: UInt16,
  _ softmaxScale: Float,
  _ causal: Bool,
  _ inputPrecision: Int32,
  _ intermediatePrecision: Int32,
  _ outputPrecision: Int32,
  _ transposeQ: Bool,
  _ transposeK: Bool,
  _ transposeV: Bool,
  _ transposeO: Bool
)
  -> Int32
{
  mfa_attention_forward(
    context,
    q, k, v, out,
    batchSize,
    seqLenQ,
    seqLenKV,
    numHeads,
    headDim,
    softmaxScale,
    causal,
    inputPrecision,
    intermediatePrecision,
    outputPrecision,
    transposeQ,
    transposeK,
    transposeV,
    transposeO,
    nil as UnsafeMutableRawPointer?,
    0,
    nil as UnsafePointer<Int64>?,
    nil as UnsafePointer<Int64>?,
    0,
    Int32(MFA_MASK_TYPE_NONE),
    Int32(MFA_MASK_SCALAR_BYTE)
  )
}

@inline(__always)
func mfa_attention_forward_str_nomask(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?,
  _ k: UnsafeMutableRawPointer?,
  _ v: UnsafeMutableRawPointer?,
  _ out: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ headDim: UInt16,
  _ softmaxScale: Float,
  _ causal: Bool,
  _ inputPrecision: UnsafePointer<CChar>?,
  _ intermediatePrecision: UnsafePointer<CChar>?,
  _ outputPrecision: UnsafePointer<CChar>?,
  _ transposeQ: Bool,
  _ transposeK: Bool,
  _ transposeV: Bool,
  _ transposeO: Bool
)
  -> Int32
{
  mfa_attention_forward_str(
    context,
    q, k, v, out,
    batchSize,
    seqLenQ,
    seqLenKV,
    numHeads,
    headDim,
    softmaxScale,
    causal,
    inputPrecision,
    intermediatePrecision,
    outputPrecision,
    transposeQ,
    transposeK,
    transposeV,
    transposeO,
    nil as UnsafeMutableRawPointer?,
    0,
    nil as UnsafePointer<Int64>?,
    nil as UnsafePointer<Int64>?,
    0,
    Int32(MFA_MASK_TYPE_NONE),
    Int32(MFA_MASK_SCALAR_BYTE)
  )
}

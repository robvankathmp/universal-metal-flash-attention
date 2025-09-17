import MFABridge
import XCTest

/// Basic test to verify FP32 precision works without NaN
final class BasicPrecisionTest: XCTestCase {
  func testFP32DoesNotProduceNaN() throws {
    // Test configuration
    let batch: UInt32 = 1
    let heads: UInt32 = 4
    let seqLen: UInt32 = 32
    let headDim: UInt16 = 16
    let elements = Int(batch * heads * seqLen * UInt32(headDim))

    // Generate test data
    var seed = 42
    var qData = [Float](repeating: 0, count: elements)
    var kData = [Float](repeating: 0, count: elements)
    var vData = [Float](repeating: 0, count: elements)

    for i in 0..<elements {
      seed = (seed &* 1_664_525 &+ 1_013_904_223) & 0xFFFF_FFFF
      qData[i] = Float(seed % 1_000_000) / 5_000_000.0 - 0.1
      seed = (seed &* 1_664_525 &+ 1_013_904_223) & 0xFFFF_FFFF
      kData[i] = Float(seed % 1_000_000) / 5_000_000.0 - 0.1
      seed = (seed &* 1_664_525 &+ 1_013_904_223) & 0xFFFF_FFFF
      vData[i] = Float(seed % 1_000_000) / 5_000_000.0 - 0.1
    }

    // Create context
    var context: UnsafeMutableRawPointer?
    let createResult = mfa_create_context(&context)
    XCTAssertEqual(createResult, 0, "Failed to create context")
    defer { mfa_destroy_context(context) }

    // Create buffers
    var q: UnsafeMutableRawPointer?
    var k: UnsafeMutableRawPointer?
    var v: UnsafeMutableRawPointer?
    var out: UnsafeMutableRawPointer?

    let bytes = elements * MemoryLayout<Float>.size

    XCTAssertEqual(mfa_create_buffer(context, bytes, &q), 0)
    defer { mfa_destroy_buffer(q) }

    XCTAssertEqual(mfa_create_buffer(context, bytes, &k), 0)
    defer { mfa_destroy_buffer(k) }

    XCTAssertEqual(mfa_create_buffer(context, bytes, &v), 0)
    defer { mfa_destroy_buffer(v) }

    XCTAssertEqual(mfa_create_buffer(context, bytes, &out), 0)
    defer { mfa_destroy_buffer(out) }

    // Copy data to buffers
    if let qContents = mfa_buffer_contents(q) {
      qData.withUnsafeBytes { ptr in
        qContents.copyMemory(from: ptr.baseAddress!, byteCount: bytes)
      }
    }

    if let kContents = mfa_buffer_contents(k) {
      kData.withUnsafeBytes { ptr in
        kContents.copyMemory(from: ptr.baseAddress!, byteCount: bytes)
      }
    }

    if let vContents = mfa_buffer_contents(v) {
      vData.withUnsafeBytes { ptr in
        vContents.copyMemory(from: ptr.baseAddress!, byteCount: bytes)
      }
    }

    // Call attention with FP32 precision using string-based API
    let result = mfa_attention_forward_str_nomask(
      context, q, k, v, out,
      batch, seqLen, seqLen, heads, headDim,
      1.0 / sqrtf(Float(headDim)), // softmax scale
      false, // not causal
      "fp32", // FP32 input precision
      "fp32", // FP32 intermediate precision
      "fp32", // FP32 output precision
      false, false, false, false // no transpose
    )

    XCTAssertEqual(result, 0, "FP32 attention failed")

    // Check output for NaN
    if let outContents = mfa_buffer_contents(out) {
      let outPointer = outContents.bindMemory(to: Float.self, capacity: elements)
      var nanCount = 0
      var nonZeroCount = 0

      for i in 0..<elements {
        if outPointer[i].isNaN {
          nanCount += 1
        }
        if abs(outPointer[i]) > 1e-8 {
          nonZeroCount += 1
        }
      }

      XCTAssertEqual(nanCount, 0, "Found \(nanCount) NaN values in FP32 output")
      XCTAssertGreaterThan(nonZeroCount, 0, "Output is all zeros")

      print("✅ FP32 test passed: \(nonZeroCount)/\(elements) non-zero values, no NaNs")
    }
  }
}

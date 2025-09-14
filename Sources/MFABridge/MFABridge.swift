import Foundation
import Metal
import FlashAttention

// MARK: - Internal Types

final class MFAContext {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init?(device: MTLDevice) {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            return nil
        }
        self.commandQueue = queue
    }
}

final class MFABuffer {
    let buffer: MTLBuffer

    init(buffer: MTLBuffer) {
        self.buffer = buffer
    }
}

// MARK: - C Bridge Implementation

@_cdecl("mfa_create_context")
public func mfa_create_context(_ context: UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32 {
    guard let device = MTLCreateSystemDefaultDevice() else {
        return 3 // MFA_ERROR_DEVICE_NOT_SUPPORTED
    }

    guard let mfaContext = MFAContext(device: device) else {
        return 2 // MFA_ERROR_MEMORY_ALLOCATION
    }

    let unmanagedContext = Unmanaged.passRetained(mfaContext)
    context.pointee = unmanagedContext.toOpaque()
    return 0 // MFA_SUCCESS
}

@_cdecl("mfa_destroy_context")
public func mfa_destroy_context(_ context: UnsafeMutableRawPointer?) {
    guard let context = context else { return }
    let unmanagedContext = Unmanaged<MFAContext>.fromOpaque(context)
    unmanagedContext.release()
}

@_cdecl("mfa_create_buffer")
public func mfa_create_buffer(
    _ context: UnsafeMutableRawPointer?,
    _ sizeBytes: Int,
    _ buffer: UnsafeMutablePointer<UnsafeMutableRawPointer?>
) -> Int32 {
    guard let context = context else { return 1 } // MFA_ERROR_INVALID_ARGS

    let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()

    guard let mtlBuffer = mfaContext.device.makeBuffer(length: sizeBytes, options: .storageModeShared) else {
        return 2 // MFA_ERROR_MEMORY_ALLOCATION
    }

    let mfaBuffer = MFABuffer(buffer: mtlBuffer)
    let unmanagedBuffer = Unmanaged.passRetained(mfaBuffer)
    buffer.pointee = unmanagedBuffer.toOpaque()

    return 0 // MFA_SUCCESS
}

@_cdecl("mfa_buffer_from_ptr")
public func mfa_buffer_from_ptr(
    _ context: UnsafeMutableRawPointer?,
    _ dataPtr: UnsafeMutableRawPointer?,
    _ sizeBytes: Int,
    _ buffer: UnsafeMutablePointer<UnsafeMutableRawPointer?>
) -> Int32 {
    guard let context = context,
          let dataPtr = dataPtr else { return 1 } // MFA_ERROR_INVALID_ARGS

    let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()

    guard let mtlBuffer = mfaContext.device.makeBuffer(
        bytes: dataPtr,
        length: sizeBytes,
        options: .storageModeShared
    ) else {
        return 2 // MFA_ERROR_MEMORY_ALLOCATION
    }

    let mfaBuffer = MFABuffer(buffer: mtlBuffer)
    let unmanagedBuffer = Unmanaged.passRetained(mfaBuffer)
    buffer.pointee = unmanagedBuffer.toOpaque()

    return 0 // MFA_SUCCESS
}

@_cdecl("mfa_buffer_contents")
public func mfa_buffer_contents(_ buffer: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let buffer = buffer else { return nil }

    let mfaBuffer = Unmanaged<MFABuffer>.fromOpaque(buffer).takeUnretainedValue()
    return mfaBuffer.buffer.contents()
}

@_cdecl("mfa_destroy_buffer")
public func mfa_destroy_buffer(_ buffer: UnsafeMutableRawPointer?) {
    guard let buffer = buffer else { return }
    let unmanagedBuffer = Unmanaged<MFABuffer>.fromOpaque(buffer)
    unmanagedBuffer.release()
}

// MARK: - Attention Functions

@_cdecl("mfa_attention_forward")
public func mfa_attention_forward(
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
) -> Int32 {
    guard let context = context,
          let q = q, let k = k, let v = v, let out = out else {
        return 1 // MFA_ERROR_INVALID_ARGS
    }

    // Extract context and buffers
    let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()
    let qBuffer = Unmanaged<MFABuffer>.fromOpaque(q).takeUnretainedValue()
    let kBuffer = Unmanaged<MFABuffer>.fromOpaque(k).takeUnretainedValue()
    let vBuffer = Unmanaged<MFABuffer>.fromOpaque(v).takeUnretainedValue()
    let outBuffer = Unmanaged<MFABuffer>.fromOpaque(out).takeUnretainedValue()

    // For now, handle single-head case (MFA's current limitation)
    // TODO: Add multi-head support by looping over heads
    if numHeads != 1 {
        return 1 // MFA_ERROR_INVALID_ARGS - Multi-head not yet supported
    }

    do {
        // Create attention descriptor
        var descriptor = AttentionDescriptor()
        descriptor.matrixDimensions = (row: seqLenQ, column: seqLenKV, head: headDim)
        descriptor.transposeState = (Q: transposeQ, K: transposeK, V: transposeV, O: transposeO)

        // Set precision based on input parameters
        descriptor.lowPrecisionInputs = (inputPrecision == 0) // FP16
        descriptor.lowPrecisionIntermediates = (intermediatePrecision == 0) // FP16

        // Create kernel descriptor
        let kernelDescriptor = descriptor.kernelDescriptor(type: .forward)
        let kernel = AttentionKernel(descriptor: kernelDescriptor)

        // Create command buffer
        guard let commandBuffer = mfaContext.commandQueue.makeCommandBuffer() else {
            return 5 // MFA_ERROR_EXECUTION_FAILED
        }

        // Set up function constants
        let constants = MTLFunctionConstantValues()
        descriptor.setFunctionConstants(constants)

        // Get the Metal function and create compute pipeline
        let source = kernel.createSource()
        let library = try mfaContext.device.makeLibrary(source: source, options: nil)
        let function = try library.makeFunction(name: "attention", constantValues: constants)
        let pipeline = try mfaContext.device.makeComputePipelineState(function: function)

        // Create compute encoder
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return 5 // MFA_ERROR_EXECUTION_FAILED
        }

        encoder.setComputePipelineState(pipeline)

        // Set buffers
        encoder.setBuffer(qBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(kBuffer.buffer, offset: 0, index: 1)
        encoder.setBuffer(vBuffer.buffer, offset: 0, index: 2)
        encoder.setBuffer(outBuffer.buffer, offset: 0, index: 3)

        // Set threadgroup memory
        encoder.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)

        // Dispatch
        let threadgroupSize = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)
        let gridSize = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)

        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        // Execute
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            print("Metal execution error: \(error)")
            return 5 // MFA_ERROR_EXECUTION_FAILED
        }

        return 0 // MFA_SUCCESS

    } catch {
        print("MFA Error: \(error)")
        return 4 // MFA_ERROR_KERNEL_COMPILATION
    }
}

// MARK: - Utility Functions

@_cdecl("mfa_error_string")
public func mfa_error_string(_ error: Int32) -> UnsafePointer<CChar>? {
    let errorString: String
    switch error {
    case 0: errorString = "Success"
    case 1: errorString = "Invalid arguments"
    case 2: errorString = "Memory allocation failed"
    case 3: errorString = "Device not supported"
    case 4: errorString = "Kernel compilation failed"
    case 5: errorString = "Execution failed"
    default: errorString = "Unknown error"
    }

    return UnsafePointer<CChar>(strdup(errorString))
}

@_cdecl("mfa_is_device_supported")
public func mfa_is_device_supported() -> Bool {
    return MTLCreateSystemDefaultDevice() != nil
}

@_cdecl("mfa_get_version")
public func mfa_get_version(_ major: UnsafeMutablePointer<Int32>?,
                           _ minor: UnsafeMutablePointer<Int32>?,
                           _ patch: UnsafeMutablePointer<Int32>?) {
    major?.pointee = 1
    minor?.pointee = 0
    patch?.pointee = 0
}